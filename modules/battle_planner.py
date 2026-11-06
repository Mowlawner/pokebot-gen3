"""Conservative, explainable battle planning.

The planner deliberately consumes normalized, planner-facing facts.  The
adapter at the bottom is the only part that knows about the live battle
objects; this keeps strategic reasoning out of battle menu input code.
"""

from dataclasses import dataclass, replace
from enum import Enum
from time import perf_counter
from typing import Any

from modules.battle_observation import (
    BattleKnowledge,
    DamageOccurred,
    KnowledgePolicy,
    PokemonFainted,
    PokemonSwitched,
)
from modules.battle_expectiminimax import (
    OpponentPolicy,
    SearchConfig,
    SearchItem,
    SearchMove,
    SearchResultCategory,
    SearchSwitch,
    search_planner_context,
)
from modules.battle_kind import BattleKind, classify_battle
from modules.console import diagnostic_print, print_always
from modules.context import context
from modules.trainer_checkpoints import known_moves_for_battle


def _search_probability_text(result, attribute: str) -> str:
    """Do not print numeric risk for a projection or unfinished search."""

    category = getattr(getattr(result, "category", None), "value", None)
    if category != "terminal_complete" or not getattr(result, "complete", False):
        return "unknown"
    value = getattr(result, attribute, None)
    return f"{value:.1%}" if isinstance(value, (int, float)) else "unknown"


def _search_category_text(result) -> str:
    return getattr(getattr(result, "category", None), "value", "unknown")


def _immediate_projection_text(result) -> str:
    """Render shallow RNG branches even when the deep result is incomplete."""

    projection = getattr(result, "immediate_projection", None)
    if projection is None:
        return "unavailable"

    def render(branch) -> str:
        probability = getattr(branch, "probability", None)
        probability_text = f"{probability:.1%}" if isinstance(probability, (int, float)) else "unknown"
        detail = f"{branch.result} p={probability_text}"
        damage = getattr(branch, "damage", 0)
        if damage:
            detail += f" damage={damage}"
        secondary = getattr(branch, "secondary_effect", None)
        if secondary:
            detail += f" status={secondary}"
        return f"{branch.action_name}:{detail}"

    player = ",".join(render(branch) for branch in getattr(projection, "player_outcomes", ())) or "none"
    opponent = ",".join(render(branch) for branch in getattr(projection, "opponent_responses", ())) or "none"
    order = ",".join(getattr(projection, "turn_order", ())) or "unknown"
    return f"player=[{player}] opponent=[{opponent}] order=[{order}]"


def _switch_options_text(result, planner_context=None) -> str:
    """Explain the expected immediate response for every switch candidate."""

    candidates = tuple(
        candidate
        for candidate in getattr(result, "root_evaluations", ())
        if getattr(candidate, "action_kind", None) == "switch"
    )
    if not candidates:
        return "none considered"

    def probability(candidate, attribute: str) -> str:
        category = getattr(getattr(candidate, "category", None), "value", None)
        if category != "terminal_complete":
            return "unknown"
        value = getattr(candidate, attribute, None)
        return f"{value:.1%}" if isinstance(value, (int, float)) else "unknown"

    options = []
    for candidate in candidates:
        line = format_search_principal_variation(candidate, planner_context)
        options.append(
            f"{candidate.action_name}: expected={candidate.expected_score:.1f}, "
            f"win={probability(candidate, 'win_probability')}, "
            f"party_wipe={probability(candidate, 'party_wipe_probability')}, "
            f"category={getattr(getattr(candidate, 'category', None), 'value', 'unknown')}, "
            f"progress={getattr(candidate, 'objective_progress', 0.0):.3f}, "
            f"projection_safe={getattr(candidate, 'projection_safe', False)!r}, "
            f"budget={getattr(candidate, 'budget_exhaustion', None)!r}, "
            f"immediate={_immediate_projection_text(candidate)}, "
            f"line={line or 'unavailable'}"
        )
    return "[" + " | ".join(options) + "]"


_SEARCH_LINE_EVENTS = frozenset(
    {
        "CAPTURED",
        "ENCOUNTER_LOST",
        "ESCAPED",
        "FAINT",
        "FAILED",
        "HORIZON",
        "KO",
        "NO_ACTION",
        "SEARCH_LIMIT",
        "STATUS_SKIP",
        "UNAVAILABLE",
        "OPPONENT_FLED",
    }
)

# These tokens are emitted by the search as player actions, rather than as
# observed move names.  In particular, a flee branch can occur after the
# opponent's response; inferring its actor by simply alternating tokens then
# produces the misleading ``opponent:Run Away`` diagnostic.
_PLAYER_ONLY_SEARCH_ACTIONS = frozenset({"Run Away"})

# Battle planning runs in the emulator's frame loop.  The generic search
# defaults remain useful for offline evaluation and readiness analysis, but a
# live action must yield quickly enough that menu input and overworld state do
# not appear frozen.  Safety is still enforced by the planner: an incomplete
# result is reported as uncertain and does not become a safe decision.
_LIVE_SEARCH_CONFIG = SearchConfig(
    opponent_policy=OpponentPolicy.CONSERVATIVE,
    # Root actions are evaluated independently. Keep the legacy live path
    # responsive; Campaign Progression has a separate, longer budget below.
    max_nodes=20_000,
    max_duration_seconds=0.50,
)

_CAMPAIGN_SEARCH_CONFIG = SearchConfig(
    opponent_policy=OpponentPolicy.CONSERVATIVE,
    # Campaign battle decisions may spend longer comparing finite resources,
    # capture odds, switches, and damage lines. The evaluator checks its
    # deadline cooperatively at every expansion, so this remains bounded while
    # giving a capture or trainer root enough time to finish its comparison.
    max_nodes=120_000,
    max_duration_seconds=3.00,
    # A voluntary rotation costs a turn and exposes the incoming member.
    # Trainer replacement prompts carry ``free_switch=True`` and remain a
    # deliberate EXP-training opportunity without this ordinary-turn cost.
    ordinary_switch_penalty=35.0,
)


def normalized_battle_objective(
    battle_state,
    *,
    capture_target: bool = False,
    capture_available: bool = False,
) -> str:
    """Return the one objective shared by the evaluator and battle display.

    Emerald's introductory wild battle is a mandatory defeat boundary. It is
    flagged as ``FirstBattle`` and occurs before capture or escape is a legal
    campaign outcome, so neither transient capture state nor the generic wild
    transit policy may downgrade it.
    """

    try:
        from modules.battle_state import BattleType

        if BattleType.FirstBattle in getattr(battle_state, "type", BattleType(0)):
            return "defeat"
    except (AttributeError, TypeError, ValueError):
        pass

    if capture_available:
        return "capture"

    intent = getattr(context, "campaign_wild_battle_intent", None)
    # A capture target with no usable ball has crossed into ordinary battle
    # policy. The target may be defeated, but capture is no longer an available
    # objective for this decision boundary.
    if capture_target and intent == "capture":
        return "defeat"
    if not getattr(battle_state, "is_trainer_battle", False):
        return intent or "defeat"
    return "defeat"


_LAST_BATTLE_KIND_LOG: tuple | None = None


def _log_battle_kind(battle_state, knowledge, planner_context) -> None:
    """Name the battle kind once per battle (and again if it changes)."""

    global _LAST_BATTLE_KIND_LOG
    try:
        from modules.battle_state import BattleType

        first_battle = BattleType.FirstBattle in getattr(battle_state, "type", BattleType(0))
    except (AttributeError, TypeError, ValueError):
        first_battle = False
    classification = classify_battle(
        objective=planner_context.battle_objective,
        is_trainer=bool(getattr(battle_state, "is_trainer_battle", False)),
        first_battle=first_battle,
        capture_target=planner_context.capture_target,
        capture_available=planner_context.capture_available,
        intent=getattr(context, "campaign_wild_battle_intent", None),
    )
    key = (
        getattr(knowledge, "battle_id", None),
        planner_context.opponent_name,
        classification.kind,
        classification.reason,
    )
    if key == _LAST_BATTLE_KIND_LOG:
        return
    _LAST_BATTLE_KIND_LOG = key
    print_always(
        lambda: (
            f"BATTLE_KIND: kind={classification.kind.value!r} "
            f"objective={planner_context.battle_objective!r} "
            f"opponent={planner_context.opponent_name!r} reason={classification.reason!r}"
        )
    )


def format_search_principal_variation(result, planner_context=None) -> str:
    """Label projected actions by side so opponent responses are legible."""

    line = tuple(getattr(result, "principal_variation", ()) or ())
    if not line:
        return ""
    player_speed = getattr(planner_context, "active_speed", None)
    opponent_speed = getattr(planner_context, "opponent_speed", None)
    player_first = (
        True
        if player_speed is None or opponent_speed is None
        else player_speed >= opponent_speed
    )
    if getattr(result, "action_kind", "move") in {"item", "switch", "run"}:
        player_first = True
    actor = "player" if player_first else "opponent"
    formatted = []
    for token in line:
        if token in _SEARCH_LINE_EVENTS:
            formatted.append(token)
            continue
        if token in _PLAYER_ONLY_SEARCH_ACTIONS:
            formatted.append(f"player:{token}")
            actor = "opponent"
            continue
        formatted.append(f"{actor}:{token}")
        actor = "opponent" if actor == "player" else "player"
    prefix = "projected " if getattr(result, "category", None) is not None and _search_category_text(result) != "terminal_complete" else ""
    return prefix + " -> ".join(formatted)


class PlannerAction(Enum):
    UseMove = "use_move"
    SwitchPokemon = "switch_pokemon"
    UseItem = "use_item"
    RunAway = "run_away"
    Wait = "wait"
    AbortForSafety = "abort_for_safety"


class PlannerConfidence(Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INSUFFICIENT = "insufficient"


class PlannerSafety(Enum):
    SAFE = "safe"
    UNSAFE_OR_UNCERTAIN = "unsafe_or_uncertain"


class PlannerRisk(Enum):
    """Qualitative risk, deliberately not a fabricated win probability."""

    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class PlannerDecisionClass(Enum):
    SAFE = "safe"
    BEST_AVAILABLE = "best_available"
    ABORT = "abort"


class PlannerEvidenceKind(Enum):
    KNOWN = "known"
    UNKNOWN = "unknown"
    DERIVED = "derived"


@dataclass(frozen=True, slots=True)
class PlannerEvidence:
    kind: PlannerEvidenceKind
    statement: str


@dataclass(frozen=True, slots=True)
class PlannerCandidate:
    """One action considered by the battle planner.

    ``available`` describes whether the action is legally selectable from the
    current battle state.  ``safety`` is deliberately separate: an attack can
    be selectable while still being unsafe, and a switch can be selectable
    while being a strategically invalid replacement (for example, a Pokémon
    with no effective move against the opponent).
    """

    action: PlannerAction
    target: Any = None
    label: str = ""
    available: bool = False
    safety: PlannerSafety = PlannerSafety.UNSAFE_OR_UNCERTAIN
    risk: PlannerRisk = PlannerRisk.CRITICAL
    reason: str = ""
    score: tuple = ()

    @property
    def is_safe(self) -> bool:
        return self.available and self.safety is PlannerSafety.SAFE


@dataclass(frozen=True, slots=True)
class PlannerDecision:
    action: PlannerAction
    target: Any = None
    confidence: PlannerConfidence = PlannerConfidence.INSUFFICIENT
    rationale: str = ""
    safety: PlannerSafety = PlannerSafety.UNSAFE_OR_UNCERTAIN
    evidence: tuple[PlannerEvidence, ...] = ()
    rejected_candidates: tuple[str, ...] = ()
    primary_rejection_reason: str | None = None
    classification: PlannerDecisionClass = PlannerDecisionClass.ABORT
    risk: PlannerRisk = PlannerRisk.CRITICAL
    candidate_evaluations: tuple[PlannerCandidate, ...] = ()
    best_safe_candidate: PlannerCandidate | None = None
    best_available_candidate: PlannerCandidate | None = None
    stochastic_result: object | None = None
    # Single-battle roots are owned by expectiminimax. Keep the default
    # aligned with that contract for decision-like callers as well.
    evaluation_method: str = "expectiminimax"

    def __post_init__(self):
        if self.classification is PlannerDecisionClass.ABORT and self.safety is PlannerSafety.SAFE:
            object.__setattr__(self, "classification", PlannerDecisionClass.SAFE)

    @property
    def is_safe_to_execute(self) -> bool:
        return self.safety is PlannerSafety.SAFE and self.confidence in (
            PlannerConfidence.HIGH,
            PlannerConfidence.MEDIUM,
        )

    @property
    def is_best_available(self) -> bool:
        return self.classification is PlannerDecisionClass.BEST_AVAILABLE


@dataclass(frozen=True, slots=True)
class PlannerMove:
    index: int
    name: str
    damage_min: int
    damage_max: int
    mechanics_supported: bool = True
    accuracy: float = 1.0
    priority: int = 0
    effect: str | None = None
    stat_target: str | None = None
    stat_delta: int = 0
    # The attacking stat is separate from ``stat_target``. The latter
    # describes a stat-stage effect, while this fact tells the stochastic
    # evaluator whether damage uses Attack or Special Attack.
    damage_stat: str | None = None
    # Gen III semi-invulnerable moves (Fly, Dig, Dive, and Bounce) spend one
    # turn charging before their damaging turn.  Keeping this on the
    # planner-facing fact lets the evaluator model the continuation without
    # changing the executor's ordinary move-selection API.
    turns_required: int = 1
    secondary_chance: float = 0.0
    # Remaining PP is carried into the stochastic search when live facts are
    # available. ``None`` keeps fixture and synthetic moves unbounded.
    pp: int | None = None


@dataclass(frozen=True, slots=True)
class PlannerOpponentMove:
    name: str
    damage_min: int
    damage_max: int
    accuracy: float = 1.0
    priority: int = 0
    mechanics_supported: bool = True
    source: str = "known"
    critical_damage_max: int | None = None
    critical_hit_chance: float = 1 / 16
    # A move revealed during battle is evidence of one possible response. It
    # is not evidence that the opponent has no other move. Direct planner
    # tests may leave this true to describe a complete synthetic response
    # set; the live adapter marks observed facts incomplete.
    response_facts_complete: bool = True
    secondary_chance: float = 0.0
    effect: str | None = None
    causes_opponent_escape: bool = False
    damage_stat: str | None = None


@dataclass(frozen=True, slots=True)
class PlannerSwitch:
    party_index: int
    name: str
    hp: int
    max_hp: int
    incoming_damage_max: int | None
    offensive_damage_max: int = 0
    incoming_critical_damage_max: int | None = None
    status: str | None = None
    moves: tuple[PlannerMove, ...] = ()
    speed: int | None = None
    escape_probability: float | None = None
    escape_guaranteed: bool = False


@dataclass(frozen=True, slots=True)
class RecoveryItem:
    item: Any
    quantity: int
    heal_amount: int = 0
    is_capture_ball: bool = False
    status_healed: frozenset[str] = frozenset()
    full_heal: bool = False
    utility_tier: int = 0

    @property
    def cures_all_status(self) -> bool:
        return "ALL" in self.status_healed

    @property
    def amount_healed(self) -> int:
        """Public wording for the HP component of this recovery item.

        ``heal_amount`` remains the storage name for compatibility with the
        original planner tests and callers.  The property makes the unified
        recovery contract read naturally at policy call sites: a status-only
        item heals ``0`` HP, while a Full Restore has full-heal semantics.
        """

        return self.heal_amount


# Compatibility name retained for existing callers and tests. New code should
# use RecoveryItem so HP and status effects travel together.
PlannerItem = RecoveryItem


@dataclass(frozen=True, slots=True)
class PlannerContext:
    active_name: str
    active_hp: int
    active_max_hp: int
    opponent_name: str
    opponent_hp: int
    moves: tuple[PlannerMove, ...] = ()
    switches: tuple[PlannerSwitch, ...] = ()
    items: tuple[PlannerItem, ...] = ()
    opponent_damage_max: int | None = None
    opponent_moves: tuple[PlannerOpponentMove, ...] = ()
    active_speed: int | None = None
    opponent_speed: int | None = None
    is_wild: bool = False
    is_trainer: bool = False
    can_switch: bool = False
    can_run: bool = False
    escape_confidence: PlannerConfidence = PlannerConfidence.INSUFFICIENT
    opponent_max_hp: int | None = None
    # Set only by the Nuzlocke/capture controller for the legal wild target.
    # These fields are deliberately opt-in so trainer and ordinary wild
    # battles retain their existing policy.
    capture_target: bool = False
    capture_available: bool = False
    capture_attempted: bool = False
    opponent_critical_damage_max: int | None = None
    active_party_index: int = 0
    # ``True`` means the response set is exhaustive for deterministic safety
    # purposes. A live battle normally has only revealed moves and therefore
    # sets this to False unless a separate threat-envelope adapter supplies a
    # complete set of possible responses.
    opponent_response_facts_complete: bool = True
    active_status: str | None = None
    active_status_healable: bool = False
    opponent_status: str | None = None
    battle_objective: str = "defeat"
    capture_success_probability: float | None = None
    capture_probability_for_state: object | None = None
    experience_reward: int = 0
    experience_values: tuple[tuple[int, float], ...] = ()
    free_switch: bool = False
    # Battle stat stages are live facts, not hypothetical planner state.  The
    # move ranges in ``PlannerMove`` already include these stages, so the
    # search uses these values to decide whether a further stat change can
    # have an effect and keeps relative changes separate from the live base.
    active_stat_stages: tuple[tuple[str, int], ...] = ()
    opponent_stat_stages: tuple[tuple[str, int], ...] = ()
    # Capture resources are objective-specific facts.  Ordinary battles keep
    # these at their defaults and do not expose ball inventory to planning or
    # diagnostics.
    capture_ball_count: int = 0
    escape_probability: float = 0.0
    escape_attempts: int | None = None
    escape_guaranteed: bool = False
    # Campaign Progression opts into the campaign-only search guards, such as
    # blocking immediate switch reversals and prioritizing a ready legal ball
    # over a setup switch.  Legacy planner callers keep their prior behavior.
    campaign_owned: bool = False
    # Observed player switch sequence for the active battle.  Campaign search
    # uses this to prevent a fresh planner invocation from immediately
    # reversing the switch selected on the preceding turn.
    switch_history: tuple[int, ...] = ()
    # Number of observed player switches since the opponent last took damage
    # or fainted. This is reconstructed from BattleKnowledge on every planner
    # call so search utility does not reset when the emulator returns to the
    # action-selection boundary.
    switch_streak: int = 0
    # Count of repeated party slots in that no-progress suffix. This exposes
    # a three-member cycle as additional utility pressure while preserving all
    # legal switch roots.
    switch_cycle_pressure: int = 0
    # Unsupported known responses remain visible as a conservative ceiling
    # instead of being silently treated as zero damage. Bide is modeled by the
    # expectiminimax state machine and therefore does not use this fallback.
    unsupported_opponent_moves: tuple[str, ...] = ()
    unsupported_response_ceiling: int = 0
    # Live Gen III Bide state is stored outside the ordinary move list. Carry
    # the ROM timer and accumulated direct damage into each search root.
    player_bide_turns: int = 0
    player_bide_damage: int = 0
    opponent_bide_turns: int = 0
    opponent_bide_damage: int = 0


class BattlePlanner:
    """Choose only actions whose safety can be explained from known facts."""

    _MAX_FORECAST_TURNS = 6
    _MIN_SEQUENCE_ACCURACY = 0.8
    # Campaign policy treats paralysis as a materially unreliable action
    # state. The value is intentionally conservative and can be replaced by
    # ROM-specific mechanics later without changing the planner contract.
    _STATUS_ACTION_RELIABILITY = {"paralysed": 0.5}
    _MIN_STAT_STAGE = -6
    _MAX_STAT_STAGE = 6
    _STAT_STAGE_ALIASES = {
        "defence": "defense",
        "special_defence": "special_defense",
    }

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

    def plan(self, context: PlannerContext) -> PlannerDecision:
        """Plan an action and retain the complete candidate evaluation.

        The original planner returned only its selected decision.  That made
        an ``ABORT`` result indistinguishable from a battle with no legal
        action, and it also hid why a switch or move had been rejected.  Keep
        selection and explanation as one operation so every caller receives
        the same action inventory.
        """

        decision = self._plan(context)
        candidates = self._evaluate_candidates(context, decision)
        autonomous = tuple(candidate for candidate in candidates if candidate.action is not PlannerAction.AbortForSafety)
        safe = tuple(candidate for candidate in autonomous if candidate.is_safe)
        available = tuple(candidate for candidate in autonomous if candidate.available)
        selected = next(
            (
                candidate
                for candidate in autonomous
                if candidate.action is decision.action and candidate.target == decision.target
            ),
            None,
        )
        # When the planner has selected a legal action, preserve that policy
        # choice as the reported best action.  If it aborted, rank the
        # remaining candidates comparatively for diagnostics only.
        best_safe = selected if selected is not None and selected.is_safe else max(
            safe, key=lambda candidate: candidate.score, default=None
        )
        best_available = selected if selected is not None and selected.available else max(
            available, key=lambda candidate: candidate.score, default=None
        )
        return replace(
            decision,
            candidate_evaluations=candidates,
            best_safe_candidate=best_safe,
            best_available_candidate=best_available,
        )

    @staticmethod
    def _root_item_is_live(context: PlannerContext, item: PlannerItem) -> bool:
        """Return whether an item root still has a concrete battle effect."""

        if item.quantity <= 0:
            return False
        if item.is_capture_ball:
            return bool(context.capture_target and context.capture_available)

        targets = (
            (context.active_hp, context.active_max_hp, context.active_status),
            *(
                (switch.hp, switch.max_hp, switch.status)
                for switch in context.switches
                if switch.hp > 0 and switch.max_hp > 0
            ),
        )
        return any(
            (item.full_heal and hp < max_hp)
            or (item.heal_amount > 0 and hp < max_hp)
            or _item_cures_status(item, status)
            for hp, max_hp, status in targets
        )

    @classmethod
    def _root_is_executable(cls, context: PlannerContext, candidate) -> bool:
        """Check a previously evaluated root against the current live facts."""

        kind = getattr(candidate, "action_kind", None)
        index = getattr(candidate, "action_index", None)
        if kind == "move":
            return any(move.index == index and cls._is_candidate(move) for move in context.moves)
        if kind == "switch":
            return bool(
                context.can_switch
                and any(
                    switch.party_index == index
                    and switch.hp > 0
                    and switch.max_hp > 0
                    and (switch.moves or switch.offensive_damage_max > 0)
                    for switch in context.switches
                )
            )
        if kind == "item":
            try:
                item = context.items[index]
            except (IndexError, TypeError):
                return False
            return cls._root_item_is_live(context, item)
        if kind == "run":
            return bool(context.can_run)
        return False

    @staticmethod
    def _stochastic_for_root(stochastic, candidate):
        """Carry one alternate root's evidence through the planner contract."""

        category = candidate.category
        return replace(
            stochastic,
            action_kind=candidate.action_kind,
            action_index=candidate.action_index,
            action_name=candidate.action_name,
            expected_score=candidate.expected_score,
            win_probability=candidate.win_probability,
            loss_probability=candidate.loss_probability,
            catastrophic_probability=candidate.catastrophic_probability,
            principal_variation=candidate.principal_variation,
            complete=category is not SearchResultCategory.INCOMPLETE,
            budget_exhaustion=candidate.budget_exhaustion,
            category=category,
            active_faint_probability=candidate.active_faint_probability,
            party_wipe_probability=candidate.party_wipe_probability,
            run_loss_probability=candidate.run_loss_probability,
            projection_safe=candidate.projection_safe,
            target_party_index=candidate.target_party_index,
            objective_failed=candidate.objective_failed,
            reason=(
                f"selected root was stale; continued with legal {candidate.action_kind} "
                f"root {candidate.action_name!r}. {stochastic.reason}"
            ),
        )

    def _expectiminimax_decision(
        self, context: PlannerContext, usable: list[PlannerMove]
    ) -> PlannerDecision:
        """Translate the search's selected root action into planner output.

        Every autonomous single-battle action is selected here.  A bounded or
        incomplete search still owns its best legal root projection; it is
        reported as ``BEST_AVAILABLE`` so callers can distinguish uncertainty
        from a safety proof without handing control to another policy.
        """

        evaluator_started_at = perf_counter()
        search_config = (
            _CAMPAIGN_SEARCH_CONFIG
            if context.campaign_owned
            else _LIVE_SEARCH_CONFIG
        )
        diagnostic_print(
            lambda: (
                "BATTLE_PLANNER_INPUT: "
                f"intent={context.battle_objective!r} active={context.active_name!r} "
                f"active_hp={context.active_hp!r}/{context.active_max_hp!r} "
                f"opponent={context.opponent_name!r} opponent_hp={context.opponent_hp!r}/"
                f"{context.opponent_max_hp!r} capture_target={context.capture_target!r} "
                f"capture_available={context.capture_available!r} "
                f"switches={tuple(switch.party_index for switch in context.switches)!r} "
                f"free_switch={context.free_switch!r} "
                f"switch_history={context.switch_history!r} "
                f"switch_streak={context.switch_streak!r} "
                f"switch_cycle_pressure={context.switch_cycle_pressure!r} "
                f"unsupported_opponent_moves={context.unsupported_opponent_moves!r}"
            ),
            trace=False,
        )
        diagnostic_print(
            lambda: (
                "BATTLE_EVALUATOR: phase='start' method='expectiminimax' "
                f"active={context.active_name!r} opponent={context.opponent_name!r} "
                f"effective_intent={context.battle_objective!r} "
                f"max_turns={search_config.max_turns!r} "
                f"max_nodes={search_config.max_nodes!r} "
                f"max_duration_seconds={search_config.max_duration_seconds!r}"
            ),
            trace=True,
            prefix="BATTLE_EVALUATOR",
        )
        stochastic = search_planner_context(context, config=search_config)
        diagnostic_print(
            lambda: (
                "BATTLE_EVALUATOR: phase='complete' method='expectiminimax' "
                f"duration_ms={(perf_counter() - evaluator_started_at) * 1000:.2f} "
                f"effective_intent={getattr(stochastic, 'objective', context.battle_objective)!r} "
                f"nodes={getattr(stochastic, 'expanded_nodes', None)!r} "
                f"complete={getattr(stochastic, 'complete', None)!r} "
                f"category={_search_category_text(stochastic)!r} "
                f"win={_search_probability_text(stochastic, 'win_probability')!r} "
                f"loss={_search_probability_text(stochastic, 'loss_probability')!r} "
                f"immediate_projection={_immediate_projection_text(stochastic)!r}"
            ),
            trace=True,
            prefix="BATTLE_EVALUATOR",
        )
        if stochastic is None:
            unsupported = tuple(move.name for move in context.moves if not move.mechanics_supported)
            return self._abort(
                "Expectiminimax could not construct a legal root action from the normalized battle facts."
                + (
                    f" Unsupported mechanics were conservatively rejected: {', '.join(unsupported)}."
                    if unsupported
                    else ""
                )
            )

        diagnostic_print(
            lambda: (
                "BATTLE_EVALUATOR_ROOT: "
                f"effective_intent={getattr(stochastic, 'objective', context.battle_objective)!r} "
                f"selected={stochastic.action_kind}:{stochastic.action_index}:{stochastic.action_name!r} "
                f"category={_search_category_text(stochastic)!r} "
                "candidates=["
                + "; ".join(
                    f"{candidate.action_kind}:{candidate.action_index}:{candidate.action_name!r} "
                    f"score={candidate.expected_score:.2f} "
                    f"category={candidate.category.value!r} "
                    f"progress={candidate.objective_progress:.3f} "
                    f"projection_safe={candidate.projection_safe!r} "
                    f"budget={candidate.budget_exhaustion!r} "
                    f"target_party_index={candidate.target_party_index!r} "
                    f"objective_failed={candidate.objective_failed!r} "
                    f"immediate={_immediate_projection_text(candidate)} "
                    f"line={format_search_principal_variation(candidate, context)}"
                    for candidate in getattr(stochastic, "root_evaluations", ())
                )
                + "]"
            ),
            trace=False,
        )

        selected_root = next(
            (
                candidate
                for candidate in getattr(stochastic, "root_evaluations", ())
                if candidate.action_kind == stochastic.action_kind
                and candidate.action_index == stochastic.action_index
            ),
            None,
        )
        if selected_root is not None and not self._root_is_executable(context, selected_root):
            alternate_roots = sorted(
                (
                    candidate
                    for candidate in getattr(stochastic, "root_evaluations", ())
                    if not (
                        candidate.action_kind == selected_root.action_kind
                        and candidate.action_index == selected_root.action_index
                    )
                    and self._root_is_executable(context, candidate)
                ),
                key=lambda candidate: (
                    candidate.expected_score,
                    int(candidate.projection_safe),
                    int(candidate.category is SearchResultCategory.TERMINAL_COMPLETE),
                ),
                reverse=True,
            )
            if alternate_roots:
                fallback = alternate_roots[0]
                diagnostic_print(
                    lambda: (
                        "BATTLE_EVALUATOR_ROOT_FALLBACK: "
                        f"stale={selected_root.action_kind}:{selected_root.action_index}:{selected_root.action_name!r} "
                        f"selected={fallback.action_kind}:{fallback.action_index}:{fallback.action_name!r}"
                    ),
                    trace=True,
                    prefix="BATTLE_EVALUATOR_ROOT_FALLBACK",
                )
                stochastic = self._stochastic_for_root(stochastic, fallback)

        safe = bool(getattr(stochastic, "is_safe", False))
        classification = PlannerDecisionClass.SAFE if safe else PlannerDecisionClass.BEST_AVAILABLE
        confidence = PlannerConfidence.HIGH if safe else PlannerConfidence.MEDIUM
        safety = PlannerSafety.SAFE if safe else PlannerSafety.UNSAFE_OR_UNCERTAIN
        risk = self._risk_from_stochastic(stochastic, safe=safe)
        rationale = (
            f"expectiminimax intent={getattr(stochastic, 'objective', context.battle_objective)!r} "
            f"selected {stochastic.action_name or 'no action'}; "
            f"category={_search_category_text(stochastic)}; "
            f"line={format_search_principal_variation(stochastic, context) or 'unavailable'}; "
            f"immediate={_immediate_projection_text(stochastic)}; "
            f"switches={_switch_options_text(stochastic, context)}; {stochastic.reason}"
        )
        if stochastic.action_kind == "move":
            selected_move = next((move for move in usable if move.index == stochastic.action_index), None)
            if selected_move is not None:
                response_damage = {
                    candidate.name: candidate.damage_max for candidate in context.opponent_moves
                }
                projected_incoming = sum(
                    response_damage.get(token, 0)
                    for token in getattr(stochastic, "principal_variation", ())
                )
                if selected_move.damage_min > 0 and context.opponent_moves:
                    minimum_hits = max(
                        1,
                        (context.opponent_hp + selected_move.damage_min - 1)
                        // selected_move.damage_min,
                    )
                    response_ceiling = max(
                        (
                            candidate.critical_damage_max
                            if candidate.critical_damage_max is not None
                            else candidate.damage_max
                            for candidate in context.opponent_moves
                        ),
                        default=0,
                    )
                    # The final KO turn ends the exchange before an ordinary
                    # response, so a multi-hit defeat line exposes the lead
                    # to at most ``hits - 1`` known responses.
                    projected_incoming = max(
                        projected_incoming,
                        response_ceiling * max(0, minimum_hits - 1),
                    )
                rationale += (
                    f" {selected_move.name}: {selected_move.damage_min}-{selected_move.damage_max} damage; "
                    f"accuracy {selected_move.accuracy:.0%}; active HP {context.active_hp}/{context.active_max_hp} HP; "
                    f"projected incoming damage {projected_incoming}."
                )
                critical_ceiling = max(
                    (
                        candidate.critical_damage_max
                        if candidate.critical_damage_max is not None
                        else candidate.damage_max
                        for candidate in context.opponent_moves
                    ),
                    default=context.opponent_critical_damage_max or 0,
                )
                if critical_ceiling:
                    rationale += (
                        f" Known critical-hit ceiling is {critical_ceiling} damage."
                    )
                if selected_move.turns_required > 1:
                    rationale += (
                        f" {selected_move.name} (charge) → {selected_move.name}; "
                        f"Two-turn moves are modeled as a charge followed by an automatic strike ({selected_move.turns_required} turn(s))."
                    )
                if selected_move.damage_min > 0:
                    rationale += (
                        f" estimated {max(1, (context.opponent_hp + selected_move.damage_min - 1) // selected_move.damage_min)} hits at minimum damage."
                    )
                setup_alternatives = tuple(
                    move.name
                    for move in usable
                    if move.index != selected_move.index
                    and (move.effect or move.stat_delta or move.damage_max == 0)
                )
                if setup_alternatives:
                    rationale += (
                        " Setup alternatives evaluated with effective stat changes modeled: "
                        + ", ".join(setup_alternatives)
                        + "."
                    )
        if not safe:
            if not self._response_facts_complete(context):
                rationale += (
                    " No guaranteed-safe sequence established because the opponent response is unknown."
                    if not context.opponent_moves and context.opponent_damage_max is None
                    else " No guaranteed-safe sequence established because the opponent response envelope is incomplete."
                )
            else:
                rationale += " No guaranteed-safe sequence established."
        elif stochastic.action_kind == "move":
            rationale += " safe sequence established."
        if (
            stochastic.action_kind == "move"
            and stochastic.principal_variation[:2] == (stochastic.action_name, "KO")
            and safe
        ):
            rationale += " guaranteed KO: the selected move reaches 0 HP before an opponent response."
        evidence = (
            PlannerEvidence(
                PlannerEvidenceKind.KNOWN,
                f"{context.active_name} HP is {context.active_hp}/{context.active_max_hp}",
            ),
            PlannerEvidence(
                PlannerEvidenceKind.KNOWN,
                f"{context.opponent_name} HP is {context.opponent_hp}/{context.opponent_max_hp or 'unknown'}",
            ),
        )
        if not self._response_facts_complete(context):
            evidence += (
                PlannerEvidence(
                    PlannerEvidenceKind.UNKNOWN,
                    "the opponent response envelope is incomplete or unknown",
                ),
            )
        evidence += (
            PlannerEvidence(PlannerEvidenceKind.DERIVED, rationale),
        )
        rejected_candidates = tuple(
            f"{candidate.action_name}: expected score {candidate.expected_score:.2f}"
            for candidate in getattr(stochastic, "root_evaluations", ())
            if not (
                candidate.action_kind == stochastic.action_kind
                and candidate.action_index == stochastic.action_index
            )
        )
        if not rejected_candidates and not safe:
            rejected_candidates = ("selected line did not establish a safety proof",)
        common = dict(
            confidence=confidence,
            rationale=rationale,
            safety=safety,
            evidence=evidence,
            rejected_candidates=rejected_candidates,
            classification=classification,
            risk=risk,
            stochastic_result=stochastic,
            evaluation_method="expectiminimax",
        )

        if stochastic.action_kind == "run":
            return PlannerDecision(PlannerAction.RunAway, None, **common)
        if stochastic.action_kind == "switch":
            replacement = next(
                (candidate for candidate in context.switches if candidate.party_index == stochastic.action_index),
                None,
            )
            if replacement is not None:
                return PlannerDecision(PlannerAction.SwitchPokemon, replacement.party_index, **common)
        if stochastic.action_kind == "item":
            try:
                item = context.items[stochastic.action_index]
            except (IndexError, TypeError):
                item = None
            if item is not None:
                safe_healing = self._safe_healing_action(context)
                target_index = getattr(stochastic, "target_party_index", None)
                if target_index is None:
                    # A bounded search can preserve the target on the
                    # matching root evaluation while the selected aggregate
                    # result is reconstructed without it.  Use that
                    # root-local evidence before falling back to the active
                    # slot; otherwise a reserve Potion is misread as a
                    # no-op on a full-HP lead and the campaign retries
                    # forever at the battle boundary.
                    selected_root = next(
                        (
                            candidate
                            for candidate in getattr(stochastic, "root_evaluations", ())
                            if candidate.action_kind == stochastic.action_kind
                            and candidate.action_index == stochastic.action_index
                        ),
                        None,
                    )
                    target_index = getattr(selected_root, "target_party_index", None)
                if target_index is None:
                    target_index = context.active_party_index
                    if safe_healing is not None and safe_healing[0] is item:
                        target_index = safe_healing[1]
                target_facts = None
                if target_index == context.active_party_index:
                    target_facts = (
                        context.active_hp,
                        context.active_max_hp,
                        context.active_status,
                    )
                else:
                    target = next(
                        (
                            switch
                            for switch in context.switches
                            if switch.party_index == target_index
                        ),
                        None,
                    )
                    if target is not None:
                        target_facts = (target.hp, target.max_hp, target.status)
                target_has_effect = bool(
                    target_facts is not None
                    and (
                        (item.full_heal and target_facts[0] < target_facts[1])
                        or (item.heal_amount > 0 and target_facts[0] < target_facts[1])
                        or _item_cures_status(item, target_facts[2])
                    )
                )
                if not item.is_capture_ball and not target_has_effect:
                    return self._abort(
                        f"Expectiminimax evaluated {item.item.name}, but party slot {target_index} "
                        "has no remaining HP or status effect for that item."
                    )
                target = (item.item, target_index)
                return PlannerDecision(PlannerAction.UseItem, target, **common)
        if stochastic.action_kind == "move":
            matching_move = next((move for move in usable if move.index == stochastic.action_index), None)
            if matching_move is not None:
                return PlannerDecision(PlannerAction.UseMove, matching_move.index, **common)
        return PlannerDecision(
            PlannerAction.AbortForSafety,
            None,
            PlannerConfidence.INSUFFICIENT,
            (
                f"Expectiminimax returned root action {stochastic.action_name!r}, but it was not executable "
                "from the normalized facts."
                + (
                    f" unsupported mechanics were conservatively rejected: {', '.join(move.name for move in context.moves if not move.mechanics_supported)}."
                    if any(not move.mechanics_supported for move in context.moves)
                    else ""
                )
            ),
            PlannerSafety.UNSAFE_OR_UNCERTAIN,
            stochastic_result=stochastic,
            evaluation_method="expectiminimax",
        )

    @staticmethod
    def _risk_from_stochastic(stochastic, *, safe: bool) -> PlannerRisk:
        """Keep catastrophic active-faint exposure visible in the summary."""

        if safe:
            return PlannerRisk.LOW
        active_faint = float(getattr(stochastic, "active_faint_probability", 0.0) or 0.0)
        party_wipe = float(getattr(stochastic, "party_wipe_probability", 0.0) or 0.0)
        run_loss = float(getattr(stochastic, "run_loss_probability", 0.0) or 0.0)
        if max(active_faint, party_wipe, run_loss) >= 0.05:
            return PlannerRisk.CRITICAL
        return PlannerRisk.HIGH

    def _plan(self, context: PlannerContext) -> PlannerDecision:
        if context.active_max_hp <= 0 or context.opponent_hp <= 0:
            return self._abort("Active or opponent HP is unavailable.")

        usable = [move for move in context.moves if self._is_candidate(move)]
        search_config = _CAMPAIGN_SEARCH_CONFIG if context.campaign_owned else _LIVE_SEARCH_CONFIG
        # A wild capture is a single-agent problem with chance, not an
        # adversarial game: answer it with the dedicated bounded solver and
        # defer to the generic search only when it cannot model the battle or
        # cannot find a plan within its faint-risk bound.
        classification = classify_battle(
            objective=context.battle_objective,
            is_trainer=context.is_trainer,
            capture_target=context.capture_target,
            capture_available=context.capture_available,
        )
        if classification.kind is BattleKind.WILD_CAPTURE:
            capture_decision = self._capture_decision(context)
            if capture_decision is not None:
                return capture_decision
        return self._expectiminimax_decision(context, usable)

    def _capture_decision(self, context: PlannerContext) -> PlannerDecision | None:
        """Decide a wild capture with the dedicated solver, or return ``None``."""

        from modules.battle_capture_planner import choose_capture_plan

        try:
            choice = choose_capture_plan(context)
        except (AttributeError, IndexError, KeyError, RuntimeError, TypeError, ValueError, ZeroDivisionError):
            return None
        if choice is None:
            return None
        plan = choice.plan
        if choice.risk_accepted and context.can_run and context.escape_confidence in (
            PlannerConfidence.HIGH,
            PlannerConfidence.MEDIUM,
        ):
            # Neither the lead nor any switch keeps the Pokemon in play within the
            # faint bound, and escape is legal: give up the encounter instead of
            # gambling a party member.  (Weighing the target's value against the
            # member at risk is a possible future refinement.)
            rejected = (plan.summary(),) + tuple(alternative.summary() for alternative in choice.alternatives)
            return PlannerDecision(
                PlannerAction.RunAway,
                None,
                context.escape_confidence,
                "capture planner: no plan, including switching, keeps the active Pokemon within the "
                f"faint bound (least risky was {plan.summary()} at "
                f"{plan.active_faint_probability:.1%} faint risk), so the encounter is abandoned.",
                PlannerSafety.SAFE,
                (
                    PlannerEvidence(
                        PlannerEvidenceKind.DERIVED,
                        "no lead or switch capture plan is within the active-faint bound; escape is legal",
                    ),
                ),
                rejected_candidates=rejected,
                classification=PlannerDecisionClass.SAFE,
                risk=PlannerRisk.LOW,
                evaluation_method="capture_planner",
            )
        rationale = (
            f"capture planner (bounded exact plan comparison): chose {plan.summary()}. "
            + (
                "RISK ACCEPTED: no plan kept the active Pokemon within the faint bound, "
                "so this is the least risky one. "
                if choice.risk_accepted
                else ""
            )
            + "Alternatives: "
            + (" | ".join(alternative.summary() for alternative in choice.alternatives) or "none")
        )
        evidence = (
            PlannerEvidence(
                PlannerEvidenceKind.KNOWN,
                f"{context.active_name} HP is {context.active_hp}/{context.active_max_hp}",
            ),
            PlannerEvidence(
                PlannerEvidenceKind.KNOWN,
                f"{context.opponent_name} HP is {context.opponent_hp}/{context.opponent_max_hp or 'unknown'}",
            ),
            PlannerEvidence(
                PlannerEvidenceKind.KNOWN,
                f"{context.capture_ball_count} capture ball(s) available",
            ),
            PlannerEvidence(PlannerEvidenceKind.DERIVED, rationale),
        )
        accepted = choice.risk_accepted
        common = dict(
            confidence=PlannerConfidence.MEDIUM if accepted else PlannerConfidence.HIGH,
            rationale=rationale,
            safety=PlannerSafety.UNSAFE_OR_UNCERTAIN if accepted else PlannerSafety.SAFE,
            evidence=evidence,
            rejected_candidates=tuple(alternative.summary() for alternative in choice.alternatives),
            classification=PlannerDecisionClass.BEST_AVAILABLE if accepted else PlannerDecisionClass.SAFE,
            risk=(
                (PlannerRisk.CRITICAL if plan.active_faint_probability >= 0.05 else PlannerRisk.HIGH)
                if accepted
                else PlannerRisk.LOW
            ),
            evaluation_method="capture_planner",
        )
        if accepted:
            from modules.battle_capture_planner import CaptureSolverResult

            # Not proven safe, but a concrete legal action the campaign gate may
            # accept for one turn; the result object declares exactly that.
            common["stochastic_result"] = CaptureSolverResult(
                plan.active_faint_probability, plan.capture_probability
            )
        if plan.switch_party_index is not None:
            return PlannerDecision(PlannerAction.SwitchPokemon, plan.switch_party_index, **common)
        if plan.weaken_turns == 0:
            return PlannerDecision(
                PlannerAction.UseItem,
                (choice.ball_item.item, context.active_party_index),
                **common,
            )
        return PlannerDecision(PlannerAction.UseMove, plan.move_index, **common)

        # The code below is retained temporarily as a reference for the
        # conservative fact checks that preceded the unified evaluator.  It is
        # unreachable by design: all live root actions now come from
        # expectiminimax above.
        if context.capture_target and context.capture_available and not context.is_trainer:
            # A ball is an observation/action boundary, never proof of a
            # capture. The observer will later emit PokemonCaptured.
            capture_item = self._capture_item(context)
            if capture_item is not None and (
                context.capture_attempted
                or context.capture_success_probability is not None
                or callable(context.capture_probability_for_state)
            ) and not (
                context.opponent_response_facts_complete
                and context.capture_success_probability is not None
            ):
                return PlannerDecision(
                    PlannerAction.UseItem,
                    capture_item,
                    PlannerConfidence.HIGH,
                    "Legal Nuzlocke capture target is in capture state; attempt capture before dealing more damage.",
                    PlannerSafety.SAFE,
                    classification=PlannerDecisionClass.SAFE,
                    risk=PlannerRisk.LOW,
                )

            # Keep probabilistically lethal moves in the stochastic search.
            # Their KO branches are encounter loss, but may still be worth
            # accepting when the remaining-ball sequence has a higher failure
            # probability.  The search owns that comparison.
            capture_probability_known = (
                context.capture_success_probability is not None
                or callable(context.capture_probability_for_state)
            )
            if not capture_probability_known:
                # Preserve the compatibility safety rule when the caller has
                # not supplied a catch evaluator. The live capture strategy
                # installs one before risk-aware search, so this branch does
                # not suppress the new probabilistic comparison in battles
                # with complete capture facts.
                usable = [move for move in usable if move.damage_max < context.opponent_hp]

        # A guaranteed KO is safe even when the opponent's response envelope
        # is complete: the opponent gets no response after the KO. Resolve
        # this terminal case before invoking the more expensive stochastic
        # evaluator.
        guaranteed = [
            move
            for move in usable
            if move.turns_required == 1
            and move.damage_min >= context.opponent_hp
            and self._effective_move_accuracy(context, move) >= 1
            and move.mechanics_supported
            and (
                not context.opponent_moves
                or self._player_acts_first(context, move, context.opponent_moves)
            )
        ]
        if (
            guaranteed
            and context.battle_objective not in {"avoid", "transit"}
            and not (context.capture_target and context.capture_available and not context.is_trainer)
        ):
            move = max(guaranteed, key=lambda candidate: (candidate.damage_min, candidate.damage_max))
            confidence = PlannerConfidence.HIGH if move.mechanics_supported else PlannerConfidence.MEDIUM
            return PlannerDecision(
                PlannerAction.UseMove,
                move.index,
                confidence,
                f"{move.name}: {move.damage_min}-{move.damage_max} damage; {context.opponent_name} has "
                f"{context.opponent_hp} HP; guaranteed KO. {context.active_name} has "
                f"{context.active_hp}/{context.active_max_hp} HP. No multi-turn survival is required, "
                f"so the move is considered safe. {self._response_note(context)}{self._live_stage_note(context)}",
                PlannerSafety.SAFE,
                (
                    PlannerEvidence(
                        PlannerEvidenceKind.KNOWN,
                        f"{context.active_name} HP is {context.active_hp}/{context.active_max_hp}",
                    ),
                    PlannerEvidence(
                        PlannerEvidenceKind.KNOWN,
                        f"{move.name} damage is estimated at {move.damage_min}-{move.damage_max}",
                    ),
                    PlannerEvidence(PlannerEvidenceKind.KNOWN, f"{context.opponent_name} HP is {context.opponent_hp}"),
                    PlannerEvidence(
                        PlannerEvidenceKind.DERIVED,
                        f"minimum damage {move.damage_min} reaches the opponent's remaining HP",
                    ),
                    PlannerEvidence(
                        PlannerEvidenceKind.UNKNOWN, "the opponent's unrevealed response is irrelevant after the KO"
                    ),
                ),
                risk=PlannerRisk.LOW,
            )

        # A complete response envelope will let expectiminimax compare escape
        # against every other legal turn.  When the envelope is unavailable,
        # an explicit avoidance intent still has a conservative legal escape
        # fallback rather than silently turning into training by attacking.
        if (
            context.battle_objective in {"avoid", "transit"}
            and context.can_run
            and context.escape_confidence in (PlannerConfidence.HIGH, PlannerConfidence.MEDIUM)
            and not context.opponent_response_facts_complete
        ):
            return PlannerDecision(
                PlannerAction.RunAway,
                None,
                context.escape_confidence,
                "campaign wild-battle intent is avoidance and the normalized response envelope is incomplete",
                PlannerSafety.SAFE,
                classification=PlannerDecisionClass.SAFE,
                risk=PlannerRisk.LOW,
                evaluation_method="campaign intent fallback",
            )

        # A complete response envelope can come from either the ROM trainer
        # catalog or a species level-up envelope for a wild encounter. Once
        # it exists, use the stochastic evaluator as the primary policy.
        if (context.is_trainer or context.is_wild) and context.opponent_response_facts_complete:
            evaluator_started_at = perf_counter()
            diagnostic_print(
                lambda: (
                    "BATTLE_EVALUATOR: phase='start' method='expectiminimax' "
                    f"active={context.active_name!r} opponent={context.opponent_name!r} "
                    f"max_turns={search_config.max_turns!r} "
                    f"max_nodes={search_config.max_nodes!r} "
                    f"max_duration_seconds={search_config.max_duration_seconds!r}"
                ),
                trace=True,
                prefix="BATTLE_EVALUATOR",
            )
            stochastic = search_planner_context(
                context,
                # Campaign Progression owns a conservative legal-response
                # envelope. RULE_BASED selects one damage-maximizing move and
                # therefore turns the search into a policy approximation
                # before expectiminimax can compare all known responses.
                config=search_config,
            )
            diagnostic_print(
                lambda: (
                    "BATTLE_EVALUATOR: phase='complete' method='expectiminimax' "
                    f"duration_ms={(perf_counter() - evaluator_started_at) * 1000:.2f} "
                    f"nodes={getattr(stochastic, 'expanded_nodes', None)!r} "
                    f"complete={getattr(stochastic, 'complete', None)!r} "
                    f"category={_search_category_text(stochastic)!r} "
                    f"win={_search_probability_text(stochastic, 'win_probability')!r} "
                    f"loss={_search_probability_text(stochastic, 'loss_probability')!r} "
                    f"immediate_projection={_immediate_projection_text(stochastic)!r}"
                ),
                trace=True,
                prefix="BATTLE_EVALUATOR",
            )
            if stochastic is not None:
                diagnostic_print(
                    lambda: (
                        "BATTLE_EVALUATOR_ROOT: "
                        f"selected={stochastic.action_kind}:{stochastic.action_index}:{stochastic.action_name!r} "
                        f"category={_search_category_text(stochastic)!r} "
                        "candidates=["
                        + "; ".join(
                            f"{candidate.action_kind}:{candidate.action_index}:{candidate.action_name!r} "
                            f"score={candidate.expected_score:.2f} "
                            f"win={'unknown' if candidate.category.value != 'terminal_complete' else f'{candidate.win_probability:.1%}'} "
                            f"party_wipe_risk={'unknown' if candidate.category.value != 'terminal_complete' else f'{candidate.catastrophic_probability:.1%}'} "
                            f"category={candidate.category.value!r} "
                            f"progress={candidate.objective_progress:.3f} "
                            f"projection_safe={candidate.projection_safe!r} "
                            f"budget={candidate.budget_exhaustion!r} "
                            f"objective_failed={candidate.objective_failed!r} "
                            f"immediate={_immediate_projection_text(candidate)} "
                            f"line={format_search_principal_variation(candidate, context)}"
                            for candidate in getattr(stochastic, "root_evaluations", ())
                        )
                        + "]"
                    ),
                    trace=True,
                    prefix="BATTLE_EVALUATOR_ROOT:",
                )
            if stochastic is not None:
                safe = stochastic.is_safe
                if stochastic.action_kind == "run":
                    escape_safe = (
                        context.battle_objective != "capture"
                        and context.escape_confidence in (PlannerConfidence.HIGH, PlannerConfidence.MEDIUM)
                        and stochastic.win_probability >= 0.90
                        and stochastic.category is SearchResultCategory.TERMINAL_COMPLETE
                    )
                    return PlannerDecision(
                        PlannerAction.RunAway,
                        None,
                        PlannerConfidence.HIGH if escape_safe else PlannerConfidence.MEDIUM,
                        (
                            f"expectiminimax selected Run Away; "
                            f"escape={_search_probability_text(stochastic, 'win_probability')}; "
                            f"category={_search_category_text(stochastic)}; "
                            f"line={format_search_principal_variation(stochastic, context)}; "
                            f"immediate={_immediate_projection_text(stochastic)}; "
                            f"switches={_switch_options_text(stochastic, context)}; "
                            f"{stochastic.reason}"
                        ),
                        PlannerSafety.SAFE if escape_safe else PlannerSafety.UNSAFE_OR_UNCERTAIN,
                        classification=PlannerDecisionClass.SAFE if escape_safe else PlannerDecisionClass.BEST_AVAILABLE,
                        risk=PlannerRisk.LOW if escape_safe else PlannerRisk.HIGH,
                        stochastic_result=stochastic,
                        evaluation_method="expectiminimax",
                    )
                if stochastic.action_kind == "switch":
                    replacement = next(
                        (
                            candidate
                            for candidate in context.switches
                            if candidate.party_index == stochastic.action_index
                        ),
                        None,
                    )
                    if replacement is not None:
                        return PlannerDecision(
                            PlannerAction.SwitchPokemon,
                            replacement.party_index,
                            PlannerConfidence.HIGH if safe else PlannerConfidence.MEDIUM,
                            (
                                f"expectiminimax selected Switch({replacement.name}); "
                                f"win={_search_probability_text(stochastic, 'win_probability')}; "
                                f"party-wipe risk={_search_probability_text(stochastic, 'catastrophic_probability')}; "
                                f"category={_search_category_text(stochastic)}; "
                                f"line={format_search_principal_variation(stochastic, context)}; "
                                f"immediate={_immediate_projection_text(stochastic)}; "
                                f"switches={_switch_options_text(stochastic, context)}; "
                                f"{stochastic.reason}"
                            ),
                            PlannerSafety.SAFE if safe else PlannerSafety.UNSAFE_OR_UNCERTAIN,
                            classification=PlannerDecisionClass.SAFE if safe else PlannerDecisionClass.BEST_AVAILABLE,
                            risk=PlannerRisk.LOW if safe else PlannerRisk.HIGH,
                            stochastic_result=stochastic,
                            evaluation_method="expectiminimax",
                        )
                if stochastic.action_kind == "item":
                    try:
                        item = context.items[stochastic.action_index]
                    except (IndexError, TypeError):
                        item = None
                    if item is not None:
                        return PlannerDecision(
                            PlannerAction.UseItem,
                            (
                                item.item,
                                getattr(stochastic, "target_party_index", None)
                                if getattr(stochastic, "target_party_index", None) is not None
                                else context.active_party_index,
                            ),
                            PlannerConfidence.HIGH if safe else PlannerConfidence.MEDIUM,
                            (
                                f"expectiminimax selected {item.item.name}; "
                                f"win={_search_probability_text(stochastic, 'win_probability')}; "
                                f"party-wipe risk={_search_probability_text(stochastic, 'catastrophic_probability')}; "
                                f"category={_search_category_text(stochastic)}; "
                                f"line={format_search_principal_variation(stochastic, context)}; "
                                f"immediate={_immediate_projection_text(stochastic)}; "
                                f"switches={_switch_options_text(stochastic, context)}; "
                                f"{stochastic.reason}"
                            ),
                            PlannerSafety.SAFE if safe else PlannerSafety.UNSAFE_OR_UNCERTAIN,
                            classification=PlannerDecisionClass.SAFE if safe else PlannerDecisionClass.BEST_AVAILABLE,
                            risk=PlannerRisk.LOW if safe else PlannerRisk.HIGH,
                            stochastic_result=stochastic,
                            evaluation_method="expectiminimax",
                        )
                matching_move = next(
                    (move for move in usable if move.index == stochastic.action_index),
                    None,
                )
                if matching_move is not None:
                    return PlannerDecision(
                        PlannerAction.UseMove,
                        matching_move.index,
                        PlannerConfidence.HIGH if safe else PlannerConfidence.MEDIUM,
                        (
                            f"expectiminimax selected {stochastic.action_name}; "
                            f"win={_search_probability_text(stochastic, 'win_probability')}; "
                            f"party-wipe risk={_search_probability_text(stochastic, 'catastrophic_probability')}; "
                            f"category={_search_category_text(stochastic)}; "
                        f"line={format_search_principal_variation(stochastic, context)}; "
                        f"immediate={_immediate_projection_text(stochastic)}; "
                        f"switches={_switch_options_text(stochastic, context)}; "
                        f"{stochastic.reason}"
                        ),
                        PlannerSafety.SAFE if safe else PlannerSafety.UNSAFE_OR_UNCERTAIN,
                        classification=PlannerDecisionClass.SAFE if safe else PlannerDecisionClass.BEST_AVAILABLE,
                        risk=PlannerRisk.LOW if safe else PlannerRisk.HIGH,
                        stochastic_result=stochastic,
                        evaluation_method="expectiminimax",
                    )
        safe_attacks = []
        failed_attacks = []
        for move in usable:
            result = self._evaluate_attack(context, move)
            if result is not None:
                safe_attacks.append((move, result))
            else:
                failed_attacks.append(self._attack_failure_reason(context, move))
        # With a known response, let the bounded search compare direct damage
        # with setup lines before selecting an attack.  With no response fact,
        # the safe attack evaluator is the only available proof path.
        if safe_attacks and not (context.opponent_moves or context.opponent_damage_max is not None):
            move, (confidence, rationale) = max(
                safe_attacks,
                key=lambda item: (
                    {PlannerConfidence.HIGH: 2, PlannerConfidence.MEDIUM: 1}.get(item[1][0], 0),
                    item[0].damage_min,
                    item[0].damage_max,
                ),
            )
            response_note = self._response_note(context)
            known = (
                PlannerEvidence(
                    PlannerEvidenceKind.KNOWN,
                    f"{context.active_name} HP is {context.active_hp}/{context.active_max_hp}",
                ),
                PlannerEvidence(PlannerEvidenceKind.KNOWN, f"{context.opponent_name} HP is {context.opponent_hp}"),
                PlannerEvidence(
                    PlannerEvidenceKind.KNOWN, f"{move.name} damage is estimated at {move.damage_min}-{move.damage_max}"
                ),
                PlannerEvidence(PlannerEvidenceKind.DERIVED, rationale),
            )
            return PlannerDecision(
                PlannerAction.UseMove,
                move.index,
                confidence,
                f"{rationale} {response_note}{self._live_stage_note(context)}",
                PlannerSafety.SAFE,
                known,
            )

        safe_switches = (
            [
                candidate
                for candidate in context.switches
                if candidate.hp > 0
                and candidate.max_hp > 0
                and candidate.incoming_damage_max is not None
                and candidate.incoming_damage_max < candidate.hp
                and (
                    candidate.incoming_critical_damage_max is None
                    or candidate.incoming_critical_damage_max < candidate.hp
                )
                # A switch is a battle replacement, not merely a way to put
                # an arbitrary party member on the field.  EXP-tag switches
                # need an explicit continuation plan and are handled by the
                # leveling strategy; they must not enter this fallback.
                and candidate.offensive_damage_max > 0
            ]
            if context.can_switch
            else []
        )
        responses_supported = all(move.mechanics_supported for move in context.opponent_moves)
        candidates = (
            self._search(context, usable)
            if (context.opponent_moves or context.opponent_damage_max is not None)
            and responses_supported
            else []
        )
        # Compare a proven attack line with a safe switch. A safe attack wins
        # because it preserves the current matchup and does not spend a
        # switch turn; a merely best-available attack yields to a validated
        # replacement instead of bypassing it.
        if candidates and safe_switches:
            _, _, search_state = max(candidates, key=lambda item: item[0])
            if not search_state["safe"]:
                candidates = []
        if candidates:
            best = max(candidates, key=lambda item: item[0])
            score, sequence, state = best
            first = sequence[0]
            safe = state["safe"]
            risk = self._risk_for_state(state, context)
            classification = PlannerDecisionClass.SAFE if safe else PlannerDecisionClass.BEST_AVAILABLE
            line = " → ".join(move.name for move in sequence)
            uncertainty = "" if safe else " No guaranteed-safe sequence established."
            risk_note = self._risk_note(state, risk, context)
            progress_note = (
                f" damage progress {state['damage_dealt']} damage; "
                f"live opponent HP {context.opponent_hp}/{context.opponent_max_hp or '?'}; "
                f"projected opponent HP after line "
                f"{state['remaining_opponent_hp']}/{context.opponent_max_hp or '?'};"
            )
            stat_note = (
                f" effective stat changes: {self._format_stat_changes(state['stat_changes'])}."
                if state["stat_changes"]
                else " no effective stat changes in the line."
            )
            rationale = (
                f"{line}: best modeled line ({'guaranteed win' if state['win'] else 'strongest available progress'});"
                f"{progress_note} projected incoming damage {state['incoming']}; {state['turns']} turn(s); "
                f"sequence accuracy {state['accuracy']:.0%}; active HP {context.active_hp}/{context.active_max_hp} HP; "
                f"projected HP {max(0, context.active_hp - state['incoming'])}/{context.active_max_hp}; "
                f"risk {risk.value.upper()}. {risk_note}{uncertainty}{stat_note}"
            )
            if state["win"] and all(move.damage_max > 0 for move in sequence):
                rationale += f" estimated {state['turns']} hits to KO."
                if safe:
                    rationale += " safe sequence established."
                else:
                    rationale += " KO is modeled under the nominal damage line, but survival is not guaranteed."
            if any(move.turns_required > 1 for move in sequence):
                rationale += " Two-turn moves are modeled as a charge followed by an automatic strike."
            if not state["known_response"]:
                rationale += (
                    " Opponent response envelope is incomplete, so survival is not claimed as safe."
                    if context.opponent_moves or context.opponent_damage_max is not None
                    else " Opponent response is unknown, so survival is not claimed as safe."
                )
            evidence = (
                PlannerEvidence(
                    PlannerEvidenceKind.KNOWN,
                    f"{context.active_name} HP is {context.active_hp}/{context.active_max_hp}",
                ),
                PlannerEvidence(PlannerEvidenceKind.KNOWN, f"candidate line is {line}"),
                PlannerEvidence(PlannerEvidenceKind.DERIVED, rationale),
            )
            if not safe:
                evidence += (
                    PlannerEvidence(
                        PlannerEvidenceKind.UNKNOWN,
                        "the line's uncertain outcome cannot be converted into a guaranteed survival claim",
                    ),
                )
            return PlannerDecision(
                PlannerAction.UseMove,
                first.index,
                PlannerConfidence.HIGH if safe else PlannerConfidence.MEDIUM,
                rationale,
                PlannerSafety.SAFE if safe else PlannerSafety.UNSAFE_OR_UNCERTAIN,
                evidence,
                tuple(failed_attacks) or (("No guaranteed-safe sequence established.",) if not safe else ()),
                None if safe else "No guaranteed-safe sequence established.",
                classification,
                risk,
            )

        if safe_switches:
            # Import lazily because the strategy package's public exports
            # include DefaultBattleStrategy, which itself imports this
            # planner. Keeping the shared evaluator out of module import time
            # avoids that package initialization cycle.
            from modules.battle_strategies.switch_evaluator import (
                SwitchCandidateFacts,
                SwitchEvaluationContext,
                evaluate_switch_candidate,
            )

            scored_switches = []
            for candidate in safe_switches:
                evaluation = evaluate_switch_candidate(
                    SwitchCandidateFacts(
                        party_index=candidate.party_index,
                        hp=candidate.hp,
                        max_hp=candidate.max_hp,
                        opponent_hp=context.opponent_hp,
                        damage_max=candidate.offensive_damage_max,
                        incoming_damage_max=candidate.incoming_damage_max,
                        incoming_critical_damage_max=candidate.incoming_critical_damage_max,
                        response_facts_known=candidate.incoming_damage_max is not None,
                        survival_probability=1.0,
                        strict_safe=True,
                    ),
                    SwitchEvaluationContext.PLANNER,
                )
                if evaluation.eligible:
                    scored_switches.append((evaluation.score, candidate, evaluation))
            if not scored_switches:
                # The explicit safe-switch gate above should make this
                # unreachable, but preserve the planner's conservative
                # behavior if a future fact adapter becomes inconsistent.
                scored_switches = [(0.0, candidate, None) for candidate in safe_switches]
            _, replacement, replacement_evaluation = max(scored_switches, key=lambda item: item[0])
            evaluation_reason = (
                f"; weighted switch score {replacement_evaluation.score:.2f}"
                if replacement_evaluation is not None
                else ""
            )
            return PlannerDecision(
                PlannerAction.SwitchPokemon,
                replacement.party_index,
                PlannerConfidence.MEDIUM,
                f"{context.active_name} is at {context.active_hp}/{context.active_max_hp} HP; "
                f"{replacement.name} is expected to survive the known response and has the best available matchup"
                f"{evaluation_reason}." + self._live_stage_note(context),
                PlannerSafety.SAFE,
                (
                    PlannerEvidence(
                        PlannerEvidenceKind.KNOWN,
                        f"{replacement.name} incoming damage is bounded below {replacement.hp} HP",
                    ),
                ),
            )

        # Unknown opponent responses prevent a multi-turn line from being
        # proven safe, but they do not make every legal attack meaningless.
        # Keep the distinction explicit: this is a comparative recommendation
        # only, never a safety claim.  A known safe switch above wins over it.
        available_attacks = [move for move in usable if move.damage_max > 0]
        if available_attacks:
            move = max(
                available_attacks,
                key=lambda candidate: (
                    candidate.damage_min * candidate.accuracy,
                    candidate.damage_min,
                    candidate.damage_max,
                    candidate.accuracy,
                    -candidate.index,
                ),
            )
            return PlannerDecision(
                PlannerAction.UseMove,
                move.index,
                PlannerConfidence.MEDIUM,
                f"{move.name}: strongest available legal attack ({move.damage_min}-{move.damage_max} damage, "
                f"{move.accuracy:.0%} accuracy). No safe winning sequence established because the opponent "
                f"response is unknown; risk {self._risk_for_unknown_attack(context, move).value.upper()}; "
                "this is the best available action, not a safety claim."
                + self._live_stage_note(context),
                PlannerSafety.UNSAFE_OR_UNCERTAIN,
                (
                    PlannerEvidence(PlannerEvidenceKind.KNOWN, f"{move.name} is a legal usable attack"),
                    PlannerEvidence(
                        PlannerEvidenceKind.KNOWN,
                        f"{move.name} damage is estimated at {move.damage_min}-{move.damage_max}",
                    ),
                    PlannerEvidence(PlannerEvidenceKind.UNKNOWN, "no human-visible opponent response is available"),
                ),
                tuple(failed_attacks),
                "No guaranteed-safe sequence established; opponent response is unknown.",
                PlannerDecisionClass.BEST_AVAILABLE,
                risk=self._risk_for_unknown_attack(context, move),
            )

        if context.capture_target and context.capture_available and self._capture_item(context) is not None:
            return PlannerDecision(
                PlannerAction.UseItem,
                self._capture_item(context),
                PlannerConfidence.MEDIUM,
                "No nonlethal damaging action is available; preserve the legal capture target and attempt a ball.",
                PlannerSafety.SAFE,
                classification=PlannerDecisionClass.SAFE,
                risk=PlannerRisk.MODERATE,
            )

        healing_action = self._safe_healing_action(context)
        if healing_action is not None:
            item, target_index, target_name, healed_hp, response_ceiling = healing_action
            return PlannerDecision(
                PlannerAction.UseItem,
                (item.item, target_index),
                PlannerConfidence.MEDIUM,
                f"{item.item.name} materially improves survival: healing {target_name} is expected to put "
                f"that Pokémon at {healed_hp} HP above the known maximum incoming damage.",
                PlannerSafety.SAFE,
                (
                    PlannerEvidence(
                        PlannerEvidenceKind.DERIVED,
                        "the active Pokémon survives this response while the item action heals a party target",
                    ),
                ),
            )

        if context.is_wild and not context.is_trainer and context.can_run:
            return (
                PlannerDecision(
                    PlannerAction.RunAway,
                    None,
                    context.escape_confidence,
                    f"Wild battle is unnecessary risk at {context.active_hp}/{context.active_max_hp} HP "
                    "and no sufficiently safe attack, switch, or healing action was established.",
                    PlannerSafety.SAFE,
                    (
                        PlannerEvidence(
                            PlannerEvidenceKind.DERIVED, "escape confidence meets the configured safe threshold"
                        ),
                    ),
                )
                if context.escape_confidence in (PlannerConfidence.HIGH, PlannerConfidence.MEDIUM)
                else self._abort("Escape is possible but its success probability is not known safely.")
            )

        reason = "No legal action has sufficient damage, survival, or visibility evidence."
        if any(not move.mechanics_supported for move in context.moves):
            reason += " unsupported mechanics were conservatively rejected."
        if failed_attacks:
            reason += " " + " ".join(failed_attacks[:2])
        return self._abort(reason, tuple(failed_attacks))

    @staticmethod
    def _response_ceiling(context: PlannerContext) -> int | None:
        if context.opponent_damage_max is None and not context.opponent_moves:
            return None
        return max(
            context.opponent_damage_max or 0,
            context.opponent_critical_damage_max or 0,
            max(
                (
                    0
                    if _is_bide_move(candidate)
                    else candidate.critical_damage_max
                    if candidate.critical_damage_max is not None
                    else candidate.damage_max
                    for candidate in context.opponent_moves
                ),
                default=0,
            ),
        )

    @staticmethod
    def _response_facts_complete(context: PlannerContext) -> bool:
        """Whether the response envelope is exhaustive enough for safety.

        Observing Tackle proves that Tackle is possible. It does not prove
        that Tackle is the opponent's only possible response. Live facts are
        therefore marked incomplete by the adapter, while direct planner
        callers can describe a complete synthetic envelope explicitly.
        """

        return bool(
            context.opponent_response_facts_complete
            and (context.opponent_moves or context.opponent_damage_max is not None)
            and all(move.response_facts_complete for move in context.opponent_moves)
        )

    @classmethod
    def _safe_healing_action(cls, context: PlannerContext):
        """Return one safe item turn, including a legal benched target."""

        if not cls._response_facts_complete(context):
            return None
        response_ceiling = cls._response_ceiling(context)
        if response_ceiling is None:
            return None
        targets = (
            (
                context.active_party_index,
                context.active_name,
                context.active_hp,
                context.active_max_hp,
                context.active_status,
                True,
            ),
            *(
                (switch.party_index, switch.name, switch.hp, switch.max_hp, switch.status, False)
                for switch in context.switches
                if switch.hp > 0 and switch.max_hp > 0
            ),
        )
        candidates = []
        for item in context.items:
            if item.quantity <= 0 or (item.heal_amount <= 0 and not item.full_heal and not item.status_healed):
                continue
            for target_index, target_name, target_hp, target_max_hp, target_status, is_active in targets:
                healed_hp = target_max_hp if item.full_heal else min(target_max_hp, target_hp + item.heal_amount)
                hp_progress = healed_hp > target_hp
                status_progress = _item_cures_status(item, target_status)
                if not hp_progress and not status_progress:
                    continue
                if is_active:
                    safe = (
                        (hp_progress and context.active_hp <= response_ceiling and healed_hp > response_ceiling)
                        or (status_progress and context.active_hp > response_ceiling)
                    )
                else:
                    # A benched target receives no opponent hit this turn, but
                    # the active Pokémon must still survive the response while
                    # the item action is being spent.
                    safe = context.active_hp > response_ceiling
                if safe:
                    missing_hp = max(0, target_max_hp - target_hp)
                    waste = missing_hp if item.full_heal else max(0, item.heal_amount - missing_hp)
                    candidates.append(
                        (
                            int(status_progress and not item.cures_all_status),
                            int(status_progress),
                            int(hp_progress),
                            -waste,
                            -item.heal_amount,
                            -target_index,
                            item,
                            target_index,
                            target_name,
                            healed_hp,
                            response_ceiling,
                        )
                    )
        if not candidates:
            return None
        selected = max(candidates, key=lambda candidate: candidate[:6])
        return selected[6:]

    def _evaluate_candidates(
        self, context: PlannerContext, decision: PlannerDecision
    ) -> tuple[PlannerCandidate, ...]:
        """Describe every action visible to the planner at this boundary.

        This is intentionally a report over the same normalized facts used by
        ``_plan``.  It does not invent opponent moves or treat emulator-only
        facts as certainty.  An action may therefore be ``available`` but
        ``unsafe`` when its outcome depends on an unknown response.
        """

        candidates: list[PlannerCandidate] = []
        for move in context.moves:
            if not self._is_candidate(move):
                if not move.mechanics_supported:
                    reason = "unsupported move mechanics"
                elif move.damage_max <= 0 and move.effect not in self._STAT_EFFECTS and move.stat_delta == 0:
                    reason = "move has no effective damaging move outcome against the observed opponent"
                else:
                    reason = "move is not usable under the planner's action model"
                candidates.append(
                    PlannerCandidate(
                        PlannerAction.UseMove,
                        move.index,
                        move.name,
                        True,
                        PlannerSafety.UNSAFE_OR_UNCERTAIN,
                        PlannerRisk.CRITICAL,
                        reason,
                        (move.damage_max, move.damage_min, move.accuracy, -move.index),
                    )
                )
                continue

            # A lethal move is still selectable by the game, but is not an
            # eligible action while a legal capture target is being preserved.
            if (
                context.capture_target
                and context.capture_available
                and not context.is_trainer
                and move.damage_max >= context.opponent_hp
                and not context.capture_attempted
            ):
                candidates.append(
                    PlannerCandidate(
                        PlannerAction.UseMove,
                        move.index,
                        move.name,
                        True,
                        PlannerSafety.UNSAFE_OR_UNCERTAIN,
                        PlannerRisk.CRITICAL,
                        "would KO the legal capture target before capture is attempted",
                        (move.damage_max, move.damage_min, move.accuracy, -move.index),
                    )
                )
                continue

            stat_effect = self._stat_effect(context, move, {})
            if stat_effect is not None and not stat_effect["effective"] and move.damage_max <= 0:
                candidates.append(
                    PlannerCandidate(
                        PlannerAction.UseMove,
                        move.index,
                        move.name,
                        True,
                        PlannerSafety.UNSAFE_OR_UNCERTAIN,
                        PlannerRisk.HIGH,
                        self._stat_effect_reason(move, stat_effect),
                        (0, 0, move.accuracy, -move.index),
                    )
                )
                continue
            if stat_effect is not None and move.damage_max <= 0:
                candidates.append(
                    PlannerCandidate(
                        PlannerAction.UseMove,
                        move.index,
                        move.name,
                        True,
                        PlannerSafety.UNSAFE_OR_UNCERTAIN,
                        PlannerRisk.MODERATE,
                        self._stat_effect_transition(move, stat_effect),
                        (0, 0, move.accuracy, -move.index),
                    )
                )
                continue

            evaluated = self._evaluate_attack(context, move)
            if evaluated is not None:
                confidence, rationale = evaluated
                safety = PlannerSafety.SAFE
                risk = PlannerRisk.LOW if confidence is PlannerConfidence.HIGH else PlannerRisk.MODERATE
                reason = rationale
            elif decision.action is PlannerAction.UseMove and decision.target == move.index and decision.is_best_available:
                safety = PlannerSafety.UNSAFE_OR_UNCERTAIN
                risk = decision.risk
                reason = decision.primary_rejection_reason or decision.rationale
            else:
                safety = PlannerSafety.UNSAFE_OR_UNCERTAIN
                risk = PlannerRisk.HIGH
                reason = self._attack_failure_reason(context, move)
            candidates.append(
                PlannerCandidate(
                    PlannerAction.UseMove,
                    move.index,
                    move.name,
                    True,
                    safety,
                    risk,
                    reason,
                    (move.damage_min * move.accuracy, move.damage_min, move.damage_max, -move.index),
                )
            )

        if not context.moves:
            candidates.append(
                PlannerCandidate(
                    PlannerAction.UseMove,
                    None,
                    "move selection",
                    False,
                    PlannerSafety.UNSAFE_OR_UNCERTAIN,
                    PlannerRisk.CRITICAL,
                    "no usable player moves were observed",
                )
            )

        for switch in context.switches:
            if not context.can_switch:
                candidates.append(
                    PlannerCandidate(
                        PlannerAction.SwitchPokemon,
                        switch.party_index,
                        f"switch to {switch.name} (slot {switch.party_index})",
                        False,
                        PlannerSafety.UNSAFE_OR_UNCERTAIN,
                        PlannerRisk.CRITICAL,
                        "switching is not legal at this battle boundary",
                    )
                )
                continue
            if switch.hp <= 0 or switch.max_hp <= 0:
                reason = "party member is fainted or has unavailable HP"
                safe = False
            elif switch.offensive_damage_max <= 0:
                reason = "replacement has no effective damaging move against the opponent"
                safe = False
            elif switch.incoming_damage_max is None or not self._response_facts_complete(context):
                reason = "opponent response facts are incomplete, so switch survival cannot be proven"
                safe = False
            elif switch.incoming_damage_max >= switch.hp:
                reason = (
                    f"known response can deal {switch.incoming_damage_max} damage to {switch.hp} HP; "
                    "replacement may faint immediately"
                )
                safe = False
            elif switch.incoming_critical_damage_max is not None and switch.incoming_critical_damage_max >= switch.hp:
                reason = (
                    f"ordinary response is survivable ({switch.incoming_damage_max} < {switch.hp}), but a "
                    f"critical-hit ceiling of {switch.incoming_critical_damage_max} can KO; "
                    "estimated critical chance is 6.25% per response"
                )
                safe = False
            else:
                reason = (
                    f"replacement survives known maximum response ({switch.incoming_damage_max} < {switch.hp}) "
                    f"and can deal up to {switch.offensive_damage_max} damage"
                )
                safe = True
            candidates.append(
                PlannerCandidate(
                    PlannerAction.SwitchPokemon,
                    switch.party_index,
                    f"switch to {switch.name} (slot {switch.party_index})",
                    True,
                    PlannerSafety.SAFE if safe else PlannerSafety.UNSAFE_OR_UNCERTAIN,
                    PlannerRisk.LOW if safe else PlannerRisk.CRITICAL,
                    reason,
                    (switch.offensive_damage_max, switch.hp, -switch.party_index),
                )
            )

        healing = [
            item
            for item in context.items
            if item.quantity > 0 and (item.heal_amount > 0 or item.full_heal or item.status_healed)
        ]
        response_ceiling = self._response_ceiling(context)
        healing_targets = (
            (
                context.active_party_index,
                context.active_name,
                context.active_hp,
                context.active_max_hp,
                True,
            ),
            *(
                (switch.party_index, switch.name, switch.hp, switch.max_hp, False)
                for switch in context.switches
                if switch.hp > 0 and switch.max_hp > 0
            ),
        )
        for item in healing:
            for target_index, target_name, target_hp, target_max_hp, is_active in healing_targets:
                healed_hp = target_max_hp if item.full_heal else min(target_max_hp, target_hp + item.heal_amount)
                if target_hp >= target_max_hp:
                    hp_progress = False
                    safe = False
                    reason = f"{target_name} is already at full HP"
                elif healed_hp <= target_hp:
                    hp_progress = False
                    safe = False
                    reason = f"healing would restore no HP to {target_name}"
                else:
                    hp_progress = True
                # The live adapter supplies status only for the active target
                # in this compatibility structure. Status-only cures remain
                # valid even when HP is already full.
                target_status = context.active_status if is_active else next(
                    (switch.status for switch in context.switches if switch.party_index == target_index), None
                )
                status_progress = _item_cures_status(item, target_status)
                if not hp_progress and not status_progress:
                    safe = False
                    if target_hp < target_max_hp:
                        reason = f"{target_name} has no condition this item can improve"
                elif response_ceiling is None or not self._response_facts_complete(context):
                    safe = False
                    reason = "opponent response facts are incomplete, so healing cannot prove survival"
                elif is_active and target_hp > response_ceiling and not status_progress:
                    safe = False
                    reason = (
                        f"active already has {target_hp} HP above the known response ceiling of "
                        f"{response_ceiling}; healing would waste this turn"
                    )
                elif is_active and healed_hp <= response_ceiling and not status_progress:
                    safe = False
                    reason = (
                        f"healing reaches {healed_hp} HP but known response can deal "
                        f"up to {response_ceiling} damage"
                    )
                elif not is_active and context.active_hp <= response_ceiling:
                    safe = False
                    reason = (
                        f"the active Pokémon has {context.active_hp} HP and may faint before "
                        f"the item can help {target_name}"
                    )
                else:
                    safe = True
                    reason = (
                        f"healing {target_name} reaches {healed_hp} HP while the active Pokémon "
                        f"clears the {response_ceiling}-damage response ceiling"
                        + (" and cures its status" if status_progress else "")
                    )
                candidates.append(
                    PlannerCandidate(
                        PlannerAction.UseItem,
                        (item.item, target_index),
                        f"use {getattr(item.item, 'name', item.item)} on {target_name}",
                        True,
                        PlannerSafety.SAFE if safe else PlannerSafety.UNSAFE_OR_UNCERTAIN,
                        PlannerRisk.LOW if safe else PlannerRisk.HIGH,
                        reason,
                        (int(hp_progress) + int(status_progress), -max(0, item.heal_amount), -target_index),
                    )
                )

        if context.capture_target and context.capture_available and not context.is_trainer:
            for item in context.items:
                if not item.is_capture_ball or item.quantity <= 0:
                    continue
                candidates.append(
                    PlannerCandidate(
                        PlannerAction.UseItem,
                        (item.item, 0),
                        f"throw {getattr(item.item, 'name', item.item)}",
                        True,
                        PlannerSafety.UNSAFE_OR_UNCERTAIN,
                        PlannerRisk.MODERATE,
                        "capture odds and weakening risk are compared by expectiminimax",
                        (item.quantity, 0),
                    )
                )

        # An unavailable escape is not a candidate.  Keeping it in the
        # ranked action set made diagnostics look as though expectiminimax
        # had evaluated an illegal node, even though the search correctly
        # omitted it when ``can_run`` was false.
        if context.is_wild and not context.is_trainer and context.can_run:
            safe = context.escape_confidence in (PlannerConfidence.HIGH, PlannerConfidence.MEDIUM)
            candidates.append(
                PlannerCandidate(
                    PlannerAction.RunAway,
                    None,
                    "flee",
                    True,
                    PlannerSafety.SAFE if safe else PlannerSafety.UNSAFE_OR_UNCERTAIN,
                    PlannerRisk.LOW if safe else PlannerRisk.HIGH,
                    "escape confidence meets the safety threshold" if safe else "escape success is uncertain",
                    (1, 0),
                )
            )
        # Manual mode is always a safe control outcome, but it is deliberately
        # excluded from best-safe/best-available autonomous action ranking.
        candidates.append(
            PlannerCandidate(
                PlannerAction.AbortForSafety,
                None,
                "manual stop",
                True,
                PlannerSafety.SAFE,
                PlannerRisk.LOW,
                "pause rather than execute an unproven autonomous action",
            )
        )
        return tuple(candidates)

    @staticmethod
    def _capture_item(context: PlannerContext):
        balls = [item for item in context.items if item.quantity > 0 and item.is_capture_ball]
        return (max(balls, key=lambda item: item.quantity).item, 0) if balls else None

    @classmethod
    def _is_candidate(cls, move: PlannerMove) -> bool:
        is_bide = str(move.name or "").strip().casefold() == "bide" or str(
            move.effect or ""
        ).strip().upper() == "BIDE"
        return move.mechanics_supported and (
            move.damage_max > 0
            or move.effect in cls._STAT_EFFECTS
            or move.stat_delta != 0
            or is_bide
        )

    def _search(self, context: PlannerContext, moves: list[PlannerMove]):
        """Small deterministic minimax-like search over our bounded action lines.

        It uses minimum player damage and maximum visible incoming damage for safety.
        Fallback ranking is comparative, not a fabricated probability estimate.
        """
        opponent_moves = context.opponent_moves
        response_available = bool(opponent_moves or context.opponent_damage_max is not None)
        # ``known_response`` is reserved for a complete threat envelope. We
        # still use visible responses to rank best-available lines, but never
        # turn those partial facts into a deterministic safety claim.
        known_response = self._response_facts_complete(context)
        incoming_max = max((m.damage_max for m in opponent_moves), default=context.opponent_damage_max or 0)
        incoming_critical_max = max(
            (m.critical_damage_max if m.critical_damage_max is not None else m.damage_max for m in opponent_moves),
            default=context.opponent_damage_max or 0,
        )
        results = []

        def visit(
            sequence,
            hp,
            opponent_hp,
            stages,
            incoming,
            critical_incoming,
            accuracy,
            turns,
            pending_move=None,
            stat_changes=(),
        ):
            if turns and opponent_hp <= 0:
                results.append(
                    (
                        self._score(
                            context,
                            sequence,
                            hp,
                            incoming,
                            critical_incoming,
                            turns,
                            accuracy,
                            True,
                            known_response,
                            context.opponent_hp - opponent_hp,
                            len(stat_changes),
                        ),
                        sequence,
                        {
                            "safe": known_response
                            and accuracy >= self._MIN_SEQUENCE_ACCURACY
                            and hp > 0
                            and critical_incoming < context.active_hp,
                            "win": True,
                            "incoming": incoming,
                            "critical_incoming": critical_incoming,
                            "turns": turns,
                            "known_response": known_response,
                            "accuracy": accuracy,
                            "damage_dealt": context.opponent_hp - opponent_hp,
                            "remaining_opponent_hp": opponent_hp,
                            "relative_stages": dict(stages),
                            "stat_changes": tuple(stat_changes),
                        },
                    )
                )
                return
            if turns >= self._MAX_FORECAST_TURNS:
                return
            # A semi-invulnerable move is represented as a two-step line.  The
            # first step is the charge and the second is the automatic strike;
            # no new player choice is introduced between them.
            choices = (
                ((pending_move, False),)
                if pending_move is not None
                else tuple((move, move.turns_required > 1) for move in moves)
            )
            for move, is_charge in choices:
                next_stages = dict(stages)
                stat_effect = self._stat_effect(context, move, next_stages)
                # A zero-power stat move which cannot change the live stage is
                # legal in the game but has no strategic value.  Do not let it
                # consume a forecast turn or win a tie against real damage.
                if not is_charge and stat_effect is not None and not stat_effect["effective"] and move.damage_max <= 0:
                    continue
                move_accuracy = self._effective_move_accuracy(context, move)
                next_accuracy = accuracy if is_charge else accuracy * move_accuracy
                damage = 0 if is_charge else self._scaled_damage(move, next_stages)
                next_opponent_hp = max(0, opponent_hp - damage)
                next_stat_changes = tuple(stat_changes)
                if not is_charge and stat_effect is not None and stat_effect["effective"]:
                    next_stages[stat_effect["key"]] = stat_effect["relative_target"]
                    next_stat_changes += (
                        (
                            move.name,
                            stat_effect["stat"],
                            stat_effect["current"],
                            stat_effect["target"],
                        ),
                    )
                # Unknown responses remain unknown; a known response is conservatively worst-case.
                player_first = self._player_acts_first(context, move, opponent_moves)
                # Fly/Dig/etc. avoid ordinary incoming damage during the
                # charge.  The strike is exposed normally according to turn
                # order, and a KO prevents the opponent's response.
                response = (
                    0
                    if is_charge
                    else incoming_max if response_available and (not player_first or next_opponent_hp > 0) else 0
                )
                critical_response = (
                    0
                    if is_charge
                    else incoming_critical_max if response_available and (not player_first or next_opponent_hp > 0) else 0
                )
                next_hp = hp - response
                next_incoming = incoming + response
                next_critical_incoming = critical_incoming + critical_response
                sequence_move = (
                    replace(move, name=f"{move.name} (charge)", damage_min=0, damage_max=0, turns_required=1)
                    if is_charge
                    else move
                )
                next_sequence = sequence + (sequence_move,)
                if next_hp <= 0:
                    results.append(
                        (
                            self._score(
                                context,
                                next_sequence,
                                next_hp,
                                next_incoming,
                                next_critical_incoming,
                                turns + 1,
                                next_accuracy,
                                False,
                                known_response,
                                context.opponent_hp - next_opponent_hp,
                                len(next_stat_changes),
                            ),
                            next_sequence,
                            {
                                "safe": False,
                                "win": False,
                                "incoming": next_incoming,
                                "critical_incoming": next_critical_incoming,
                                "turns": turns + 1,
                                "known_response": known_response,
                                "accuracy": next_accuracy,
                                "damage_dealt": context.opponent_hp - next_opponent_hp,
                                "remaining_opponent_hp": next_opponent_hp,
                                "relative_stages": dict(next_stages),
                                "stat_changes": tuple(next_stat_changes),
                            },
                        )
                    )
                elif next_opponent_hp <= 0:
                    results.append(
                        (
                            self._score(
                                context,
                                next_sequence,
                                next_hp,
                                next_incoming,
                                next_critical_incoming,
                                turns + 1,
                                next_accuracy,
                                True,
                                known_response,
                                context.opponent_hp - next_opponent_hp,
                                len(next_stat_changes),
                            ),
                            next_sequence,
                            {
                                "safe": known_response
                                and next_accuracy >= self._MIN_SEQUENCE_ACCURACY
                                and next_hp > 0
                                and next_critical_incoming < context.active_hp,
                                "win": True,
                                "incoming": next_incoming,
                                "critical_incoming": next_critical_incoming,
                                "turns": turns + 1,
                                "known_response": known_response,
                                "accuracy": next_accuracy,
                                "damage_dealt": context.opponent_hp - next_opponent_hp,
                                "remaining_opponent_hp": next_opponent_hp,
                                "relative_stages": dict(next_stages),
                                "stat_changes": tuple(next_stat_changes),
                            },
                        )
                    )
                else:
                    visit(
                        next_sequence,
                        next_hp,
                        next_opponent_hp,
                        next_stages,
                        next_incoming,
                        next_critical_incoming,
                        next_accuracy,
                        turns + 1,
                        move if is_charge else None,
                        next_stat_changes,
                    )

        visit((), context.active_hp, context.opponent_hp, {}, 0, 0, 1.0, 0)
        return results

    @classmethod
    def _stage_map(cls, stages: tuple[tuple[str, int], ...] | dict[str, int]) -> dict[str, int]:
        """Normalize emulator spelling and clamp live planner stage facts."""

        values = dict(stages)
        return {
            cls._STAT_STAGE_ALIASES.get(stat, stat): max(
                cls._MIN_STAT_STAGE,
                min(cls._MAX_STAT_STAGE, int(value)),
            )
            for stat, value in values.items()
        }

    @classmethod
    def _stat_effect(cls, context: PlannerContext, move: PlannerMove, relative_stages: dict):
        """Describe a move's effective stat change from the current battle state.

        ``relative_stages`` contains only changes forecast within this search.
        The damage ranges already include the live stage read from the
        emulator, so applying a relative multiplier keeps the current stage
        from being counted twice.
        """

        effect = cls._STAT_EFFECTS.get(move.effect)
        if effect is None and move.stat_delta:
            effect = ("self", move.stat_target or "attack", move.stat_delta)
        if effect is None:
            return None
        side, stat, delta = effect
        live = cls._stage_map(context.active_stat_stages if side == "self" else context.opponent_stat_stages)
        key = (side, stat)
        relative_before = relative_stages.get(key, 0)
        current = max(cls._MIN_STAT_STAGE, min(cls._MAX_STAT_STAGE, live.get(stat, 0) + relative_before))
        target = max(cls._MIN_STAT_STAGE, min(cls._MAX_STAT_STAGE, current + delta))
        return {
            "key": key,
            "stat": stat,
            "delta": delta,
            "current": current,
            "target": target,
            "effective": target != current,
            "relative_target": target - live.get(stat, 0),
        }

    @staticmethod
    def _stat_effect_reason(move: PlannerMove, effect: dict) -> str:
        direction = "lower" if effect["delta"] < 0 else "raise"
        return (
            f"{move.name}: cannot {direction} {effect['stat']} beyond stage {effect['current']}; "
            "this stat change would have no battle effect"
        )

    @staticmethod
    def _stat_effect_transition(move: PlannerMove, effect: dict) -> str:
        return (
            f"{move.name}: modeled {effect['stat']} change "
            f"{effect['current']:+d}→{effect['target']:+d} can improve the line"
        )

    @classmethod
    def _format_stat_stages(cls, stages: tuple[tuple[str, int], ...] | dict[str, int]) -> str:
        values = cls._stage_map(stages)
        if not values:
            return "none observed"
        return ", ".join(f"{stat}={value:+d}" for stat, value in sorted(values.items()))

    @classmethod
    def _format_stat_changes(cls, changes: tuple[tuple[str, str, int, int], ...]) -> str:
        if not changes:
            return "none"
        return ", ".join(f"{move} {stat} {before:+d}→{after:+d}" for move, stat, before, after in changes)

    @classmethod
    def _live_stage_note(cls, context: PlannerContext) -> str:
        return (
            f" Live stages: active ({cls._format_stat_stages(context.active_stat_stages)}); "
            f"opponent ({cls._format_stat_stages(context.opponent_stat_stages)})."
        )

    @staticmethod
    def _stage_multiplier(stage: int) -> float:
        return (2 + stage) / 2 if stage >= 0 else 2 / (2 - stage)

    def _scaled_damage(self, move: PlannerMove, stages: dict) -> int:
        if move.damage_min <= 0:
            return 0
        # damage_min is already produced by the shared type/STAB/stat model. Stat
        # effects only apply a bounded Gen III stage multiplier on top of it.
        attack_stat = move.damage_stat or move.stat_target or "attack"
        attack_stage = stages.get(("self", attack_stat), 0)
        defence_stat = "special_defense" if attack_stat == "special_attack" else "defense"
        defence_stage = stages.get(("opponent", defence_stat), 0)
        value = move.damage_min * self._stage_multiplier(attack_stage) / self._stage_multiplier(defence_stage)
        return max(0, int(value))

    @classmethod
    def _risk_for_state(cls, state, context):
        if not state["known_response"]:
            return PlannerRisk.HIGH
        if not state["win"] or state.get("safe") is False and state["incoming"] >= context.active_hp:
            return PlannerRisk.CRITICAL if state["incoming"] >= context.active_hp else PlannerRisk.HIGH
        if state.get("critical_incoming", state["incoming"]) >= context.active_hp:
            return PlannerRisk.CRITICAL
        if state.get("accuracy", 1.0) < cls._MIN_SEQUENCE_ACCURACY:
            return PlannerRisk.HIGH
        if state.get("accuracy", 1.0) < 1.0:
            return PlannerRisk.MODERATE
        return PlannerRisk.LOW

    @staticmethod
    def _risk_for_unknown_attack(context, move):
        if move.accuracy < 0.8:
            return PlannerRisk.HIGH
        return PlannerRisk.HIGH if context.opponent_hp > move.damage_min else PlannerRisk.MODERATE

    @staticmethod
    def _risk_note(state, risk, context):
        if risk is PlannerRisk.CRITICAL:
            if state.get("critical_incoming", state["incoming"]) >= context.active_hp:
                probability = state.get("critical_probability")
                if probability is None:
                    probability = BattlePlanner._critical_exposure_probability(context, state.get("turns", 0))
                probability_note = (
                    f" The modeled critical-only lethal probability is approximately {probability:.1%}."
                    if probability is not None
                    else ""
                )
                return f"A modeled critical-hit ceiling can be lethal before the line completes.{probability_note}"
            return "The modeled response can KO before the line completes."
        if risk is PlannerRisk.HIGH:
            if state.get("accuracy", 1.0) < BattlePlanner._MIN_SEQUENCE_ACCURACY:
                return "The required sequence accuracy is too uncertain for a safety claim."
            return "The line depends on surviving a meaningful opposing response."
        if risk is PlannerRisk.MODERATE:
            return "Accuracy or action order leaves bounded uncertainty."
        return "No modeled response reaches lethal damage before the win."

    @classmethod
    def _score(
        cls,
        context,
        sequence,
        hp,
        incoming,
        critical_incoming,
        turns,
        accuracy,
        win,
        known_response,
        damage_dealt,
        effective_stat_changes,
    ):
        # Lexicographic priorities mirror the Nuzlocke policy. These are ranks,
        # not probabilities. Risk is included after the fundamental survival and
        # win objectives, so a safer winning line outranks a more fragile one.
        risk = cls._risk_for_state(
            {
                "known_response": known_response,
                "win": win,
                "incoming": incoming,
                "critical_incoming": critical_incoming,
                "accuracy": accuracy,
            },
            context,
        )
        critical_probability = cls._critical_exposure_probability(context, max(0, turns))
        risk_rank = {
            PlannerRisk.LOW: 3,
            PlannerRisk.MODERATE: 2,
            PlannerRisk.HIGH: 1,
            PlannerRisk.CRITICAL: 0,
        }[risk]
        survived = hp > 0
        # Damage progress is a meaningful fallback signal.  Setup remains
        # eligible and can still win when it produces a shorter, safer, or
        # more damaging line, but a zero-damage setup sequence must not beat a
        # line that actually advances the opponent toward a KO.
        return (
            int(survived),
            int(win),
            risk_rank,
            int(hp),
            -incoming,
            -critical_probability,
            -turns,
            accuracy,
            int(damage_dealt),
            -int(effective_stat_changes),
            -sum(m.index for m in sequence),
        )

    def _evaluate_attack(self, context: PlannerContext, move: PlannerMove):
        if not move.mechanics_supported or move.damage_min <= 0 or move.accuracy <= 0:
            return None
        hits = (context.opponent_hp + move.damage_min - 1) // move.damage_min
        action_turns = hits * max(1, move.turns_required)
        if hits <= 0 or action_turns > self._MAX_FORECAST_TURNS:
            return None
        sequence_accuracy = self._effective_move_accuracy(context, move) ** hits
        if sequence_accuracy < self._MIN_SEQUENCE_ACCURACY:
            return None
        opponent_moves = context.opponent_moves
        if not opponent_moves and context.opponent_damage_max is not None:
            opponent_moves = (PlannerOpponentMove("known response", 0, context.opponent_damage_max),)
        if hits > 1 and not opponent_moves:
            return None
        # A non-guaranteed line needs a complete response envelope. The
        # guaranteed-KO path above remains valid because the opponent cannot
        # respond after the KO, even when its moves are unknown.
        if not self._response_facts_complete(context):
            return None
        if any(not candidate.mechanics_supported for candidate in opponent_moves):
            return None
        incoming = max((candidate.damage_max for candidate in opponent_moves), default=0)
        critical_incoming = max(
            (
                candidate.critical_damage_max if candidate.critical_damage_max is not None else candidate.damage_max
                for candidate in opponent_moves
            ),
            default=incoming,
        )
        player_first = self._player_acts_first(context, move, opponent_moves)
        # Semi-invulnerable moves expose the user only on their strike turn;
        # the charge turn is still part of the forecast horizon but does not
        # receive the ordinary visible response damage.
        incoming_exchanges = max(0, hits - 1) if player_first else hits
        projected_incoming = incoming * incoming_exchanges
        projected_critical_incoming = critical_incoming * incoming_exchanges
        critical_probability = self._critical_exposure_probability(context, incoming_exchanges)
        if context.active_hp <= projected_incoming or context.active_hp <= projected_critical_incoming:
            return None
        order_note = "player acts first" if player_first else "opponent may act before each attack"
        confidence = (
            PlannerConfidence.HIGH
            if move.accuracy >= 1 and context.active_speed is not None
            else PlannerConfidence.MEDIUM
        )
        return confidence, (
            f"{move.name}: {move.damage_min}-{move.damage_max} damage; estimated {hits} hits to KO; "
            f"{action_turns} action turns; sequence accuracy {sequence_accuracy:.0%}; "
            f"projected incoming damage {projected_incoming}; "
            f"{order_note}; critical-hit ceiling {projected_critical_incoming}; "
            f"critical exposure probability {critical_probability:.1%}; safe sequence established."
        )

    @classmethod
    def _effective_move_accuracy(cls, context: PlannerContext, move: PlannerMove) -> float:
        reliability = cls._STATUS_ACTION_RELIABILITY.get(context.active_status or "", 1.0)
        return max(0.0, min(1.0, move.accuracy * reliability))

    @staticmethod
    def _player_acts_first(
        context: PlannerContext, move: PlannerMove, opponent_moves: tuple[PlannerOpponentMove, ...]
    ) -> bool:
        opponent_priority = max((candidate.priority for candidate in opponent_moves), default=0)
        if move.priority != opponent_priority:
            return move.priority > opponent_priority
        if context.active_speed is not None and context.opponent_speed is not None:
            return context.active_speed > context.opponent_speed
        return False

    @staticmethod
    def _critical_exposure_probability(context, response_count: int) -> float:
        """Approximate the chance of at least one critical response.

        Damage ceilings remain conservative because the planner does not know
        the opponent's move choice or full damage-roll distribution. It can
        still account for Gen III's standard 1/16 critical-hit chance when
        comparing otherwise similar lines.
        """

        if response_count <= 0 or not context.opponent_moves:
            return 0.0
        per_response = max(
            (
                max(0.0, min(1.0, candidate.critical_hit_chance))
                for candidate in context.opponent_moves
                if candidate.damage_max > 0
            ),
            default=0.0,
        )
        return 1 - (1 - per_response) ** response_count

    @staticmethod
    def _attack_failure_reason(context: PlannerContext, move: PlannerMove) -> str:
        if not move.mechanics_supported:
            return f"{move.name}: unsupported mechanics materially reduce confidence."
        stat_effect = BattlePlanner._stat_effect(context, move, {})
        if stat_effect is not None and not stat_effect["effective"] and move.damage_max <= 0:
            return BattlePlanner._stat_effect_reason(move, stat_effect)
        if stat_effect is not None and move.damage_max <= 0:
            return (
                f"{BattlePlanner._stat_effect_transition(move, stat_effect)}; "
                "no safe winning sequence established"
            )
        if move.damage_max <= 0:
            return f"{move.name}: no damaging progress and no effective stat change is modeled."
        hits = (context.opponent_hp + move.damage_min - 1) // move.damage_min if move.damage_min > 0 else 0
        action_turns = hits * max(1, move.turns_required)
        if action_turns > BattlePlanner._MAX_FORECAST_TURNS:
            return (
                f"{move.name}: requires {action_turns} action turns for at least {hits} minimum-damage hits, "
                "beyond the forecast horizon."
            )
        effective_accuracy = BattlePlanner._effective_move_accuracy(context, move)
        if effective_accuracy < 1 and effective_accuracy ** max(1, hits) < BattlePlanner._MIN_SEQUENCE_ACCURACY:
            return f"{move.name}: accuracy/status reliability is too uncertain for the estimated {hits}-hit sequence."
        if hits > 1 and not context.opponent_moves and context.opponent_damage_max is None:
            return f"{move.name}: requires {hits} hits, but the opponent response is unknown; safe survival cannot be established."
        opponent_moves = context.opponent_moves
        incoming = max((candidate.damage_max for candidate in opponent_moves), default=context.opponent_damage_max or 0)
        player_first = BattlePlanner._player_acts_first(context, move, opponent_moves)
        incoming_exchanges = max(0, hits - 1) if player_first else hits
        projected_incoming = incoming * incoming_exchanges
        critical_incoming = (
            max(
                (
                    candidate.critical_damage_max if candidate.critical_damage_max is not None else candidate.damage_max
                    for candidate in opponent_moves
                ),
                default=incoming,
            )
            * incoming_exchanges
        )
        if (opponent_moves or context.opponent_damage_max is not None) and context.active_hp <= projected_incoming:
            return (
                f"{move.name}: {context.active_hp}/{context.active_max_hp} HP cannot survive the "
                f"projected {projected_incoming} incoming damage over {action_turns} action turns; "
                "the sequence is rejected."
            )
        if (opponent_moves or context.opponent_damage_max is not None) and context.active_hp <= critical_incoming:
            return (
                f"{move.name}: a modeled critical-hit ceiling of {critical_incoming} damage can be lethal "
                f"before the {action_turns}-turn line completes; the sequence is risky rather than safe."
            )
        return f"{move.name}: no safe winning sequence established."

    @staticmethod
    def _response_note(context: PlannerContext) -> str:
        if not context.opponent_moves:
            return "Opponent response is unknown; no legal opponent action is currently visible."
        descriptions = []
        for candidate in context.opponent_moves:
            descriptions.append(
                f"{candidate.name} is a {candidate.source} legal response ({candidate.damage_min}-{candidate.damage_max} damage)"
            )
        completeness = "complete" if BattlePlanner._response_facts_complete(context) else "incomplete"
        return f"Opponent possibilities ({completeness} envelope): " + "; ".join(descriptions) + "."

    @staticmethod
    def _abort(reason: str, rejected_candidates: tuple[str, ...] = ()) -> PlannerDecision:
        return PlannerDecision(
            PlannerAction.AbortForSafety,
            None,
            PlannerConfidence.INSUFFICIENT,
            reason,
            PlannerSafety.UNSAFE_OR_UNCERTAIN,
            (PlannerEvidence(PlannerEvidenceKind.UNKNOWN, "a safe action could not be proven from visible facts"),),
            rejected_candidates,
            rejected_candidates[0] if rejected_candidates else reason,
            PlannerDecisionClass.ABORT,
        )


def format_planner_decision_diagnostic(decision: PlannerDecision, context: PlannerContext) -> str:
    """Return one concise, evidence-labelled explanation of a decision."""
    move = next((candidate for candidate in context.moves if candidate.index == decision.target), None)
    selected = move.name if move is not None else str(decision.target)
    damage = f"{move.damage_min}-{move.damage_max}" if move else "unknown"
    hits = (context.opponent_hp + move.damage_min - 1) // move.damage_min if move and move.damage_min > 0 else None
    known_moves = [candidate.name for candidate in context.opponent_moves]
    response_complete = BattlePlanner._response_facts_complete(context)
    unknown = not response_complete
    setup = bool(move and (move.damage_max == 0 or move.effect or move.stat_delta or move.turns_required > 1))
    stat_effect = BattlePlanner._stat_effect(context, move, {}) if move is not None else None
    if stat_effect is None:
        move_effectiveness = "none selected" if move is None else "damaging or non-stat move"
    elif stat_effect["effective"]:
        move_effectiveness = (
            f"{stat_effect['stat']} {stat_effect['current']:+d}->{stat_effect['target']:+d}; effective=True"
        )
    else:
        move_effectiveness = (
            f"{stat_effect['stat']} remains {stat_effect['current']:+d}; effective=False (stage limit)"
        )
    move_effectiveness_inventory = []
    for candidate_move in context.moves:
        candidate_effect = BattlePlanner._stat_effect(context, candidate_move, {})
        if candidate_effect is None:
            continue
        move_effectiveness_inventory.append(
            f"{candidate_move.name}:{candidate_effect['current']:+d}->{candidate_effect['target']:+d};"
            f"effective={candidate_effect['effective']}"
        )
    rejected = decision.rejected_candidates[0] if decision.rejected_candidates else "none recorded"
    fallback = decision.classification is PlannerDecisionClass.BEST_AVAILABLE and not decision.is_safe_to_execute
    reason = decision.rationale
    if decision.classification is PlannerDecisionClass.BEST_AVAILABLE:
        reason = "No fully safe line established; selecting best available legal action. " + reason
    decision_line = (
        f"BATTLE_DECISION_DETAIL: {decision.classification.name}: {selected}; action={decision.action.value}; "
        f"evaluation_method={decision.evaluation_method!r}; "
        f"effective_intent={context.battle_objective!r}; "
        f"confidence={decision.confidence.value}; player_hp={context.active_hp}/{context.active_max_hp}; "
        f"opponent_hp={context.opponent_hp}/{context.opponent_max_hp or 'unknown'}; damage={damage}; "
        f"hits_to_ko={hits if hits is not None else 'n/a'}; known_opponent_moves={known_moves!r}; "
        f"active_stat_stages={BattlePlanner._format_stat_stages(context.active_stat_stages)!r}; "
        f"opponent_stat_stages={BattlePlanner._format_stat_stages(context.opponent_stat_stages)!r}; "
        f"selected_move_effectiveness={move_effectiveness!r}; "
        f"stat_move_effectiveness={move_effectiveness_inventory!r}; "
                f"opponent_response_unknown={unknown}; response_facts_complete={response_complete}; "
        f"selected_item_target={decision.target[1] if decision.action is PlannerAction.UseItem and isinstance(decision.target, tuple) else 'n/a'}; "
        f"risk={decision.risk.value}; setup_line={setup}; "
        f"fallback={fallback}; primary_reason={reason}; strongest_rejected_alternative={rejected}"
    )
    return decision_line + "\n" + format_action_evaluation_diagnostic(decision)


def format_action_evaluation_diagnostic(decision: PlannerDecision) -> str:
    """Format the full action inventory and its safety explanations.

    The line is intentionally deterministic and compact enough for a trace,
    while retaining the distinction between an action that cannot be selected,
    one that can be selected but is unsafe, and one that is safe.
    """

    def describe(candidate: PlannerCandidate) -> str:
        if not candidate.available:
            status = "unavailable"
        elif candidate.is_safe:
            status = "safe"
        else:
            status = "unsafe/uncertain"
        return f"{candidate.label}={status} ({candidate.reason})"

    best_safe = decision.best_safe_candidate
    best_available = decision.best_available_candidate
    safe_label = best_safe.label if best_safe is not None else "none"
    available_label = best_available.label if best_available is not None else "none"
    candidates = "; ".join(describe(candidate) for candidate in decision.candidate_evaluations)
    return (
        f"BATTLE_ACTION_EVALUATION: best_safe={safe_label}; best_available={available_label}; "
        f"candidates=[{candidates}]"
    )


def knowledge_for_planner(knowledge: BattleKnowledge | None) -> BattleKnowledge | None:
    """Reject emulator-only opponent facts at the planner boundary."""
    if knowledge is not None and knowledge.policy is KnowledgePolicy.FULL_EMULATOR:
        emulator_only = set(knowledge.emulator_only_opponent_moves)
        # A FULL_EMULATOR knowledge object created by older callers has no
        # quarantine index; its known opponent moves were all read from the
        # emulator and must therefore remain hidden at this boundary.
        visible_known = (
            tuple(item for item in knowledge.known_opponent_moves if item not in emulator_only) if emulator_only else ()
        )
        return BattleKnowledge(
            battle_id=knowledge.battle_id,
            policy=KnowledgePolicy.HUMAN_KNOWLEDGE,
            observations=knowledge.observations,
            events=knowledge.events,
            opponent_pokemon=knowledge.opponent_pokemon,
            observed_opponent_moves=knowledge.observed_opponent_moves,
            known_opponent_moves=visible_known,
            inferred_opponent_moves=knowledge.inferred_opponent_moves,
            revealed_opponent_moves=knowledge.revealed_opponent_moves,
            hp_history=knowledge.hp_history,
            damage_history=knowledge.damage_history,
            healing_history=knowledge.healing_history,
            status_history=knowledge.status_history,
            stat_stage_history=knowledge.stat_stage_history,
            switch_history=knowledge.switch_history,
            faint_history=knowledge.faint_history,
            outcome=knowledge.outcome,
            metadata=knowledge.metadata,
            action_history=knowledge.action_history,
            observed_player_moves=knowledge.observed_player_moves,
        )
    return knowledge


def _observed_switch_pressure(knowledge: BattleKnowledge | None) -> tuple[int, int]:
    """Return no-progress switch streak and repeated-slot pressure.

    The battle planner is called again after every accepted action, so a
    switch count stored only in SearchState disappears at the next root. Use
    authoritative battle events to find the suffix since the opponent last
    took damage or fainted. Unknown or missing events contribute no pressure;
    the planner must not invent a switch cycle from stale state.
    """

    if knowledge is None:
        return 0, 0
    events = tuple(getattr(knowledge, "events", ()) or ())
    progress_index = -1
    for index, event in enumerate(events):
        if isinstance(event, DamageOccurred) and event.side == "opponent" and event.amount > 0:
            progress_index = index
        elif isinstance(event, PokemonFainted) and event.side == "opponent":
            progress_index = index
    switches = tuple(
        event
        for event in events[progress_index + 1 :]
        if isinstance(event, PokemonSwitched) and event.side == "player"
    )
    seen_slots: set[int] = set()
    repeated_slots = 0
    for event in switches:
        for slot in (event.before_party_index, event.after_party_index):
            if slot is None:
                continue
            if slot == event.after_party_index and slot in seen_slots:
                repeated_slots += 1
            seen_slots.add(slot)
    return len(switches), repeated_slots


def opponent_response_move_names(
    battle_state: Any,
    knowledge: BattleKnowledge | None = None,
    *,
    include_hidden_live_moves: bool = False,
) -> tuple[str, ...]:
    """Return the response envelope used by battle safety policies.

    This is the single move-name adapter for both the general planner and the
    Nuzlocke emergency gate.  Human-facing planning uses revealed moves plus
    the legal level-up envelope for wild Pokémon.  The emergency gate may opt
    into the live ROM move list because it is a loss-prevention guard, while
    the planner itself continues to avoid treating hidden emulator facts as
    player knowledge.
    """

    visible = knowledge_for_planner(knowledge)
    names = set()
    if visible is not None:
        names.update(name for _, name in visible.revealed_opponent_moves if name)
        names.update(name for _, name in visible.known_opponent_moves if name)

    opponent = getattr(getattr(battle_state, "opponent", None), "active_battler", None)
    is_trainer = bool(getattr(battle_state, "is_trainer_battle", False))
    if not is_trainer:
        names.update(_wild_level_up_move_names(opponent))

    if include_hidden_live_moves and opponent is not None:
        try:
            names.update(
                learned_move.move.name
                for learned_move in opponent.moves
                if learned_move is not None
                and getattr(learned_move, "move", None) is not None
                and getattr(learned_move, "pp", 0) > 0
            )
        except (AttributeError, TypeError, ValueError, IndexError):
            pass
    return tuple(sorted(name for name in names if name))


def _wild_level_up_move_names(opponent: Any) -> set[str]:
    """Return the legal level-up threat envelope for one wild battler."""

    if opponent is None:
        return set()
    try:
        level = int(getattr(opponent, "level", 0) or 0)
        learnset = getattr(getattr(opponent.species, "learnset", None), "level_up", ())
        return {
            entry.move.name
            for entry in learnset
            if int(getattr(entry, "level", 0) or 0) <= level
            and getattr(getattr(entry, "move", None), "name", None)
        }
    except (AttributeError, TypeError, ValueError):
        return set()


def _status_value(status: Any) -> str | None:
    value = getattr(status, "value", status)
    return value if isinstance(value, str) else None


def _bide_facts(battler: Any) -> tuple[int, int]:
    """Read the optional live Bide timer and stored damage."""

    try:
        turns = max(0, int(getattr(battler, "bide_turns_remaining", 0) or 0))
    except (AttributeError, RuntimeError, TypeError, ValueError):
        turns = 0
    try:
        damage = max(0, int(getattr(battler, "bide_damage", 0) or 0))
    except (AttributeError, RuntimeError, TypeError, ValueError):
        damage = 0
    return turns, damage


def _normalized_status(status: Any) -> str:
    value = getattr(status, "value", status)
    normalized = str(value or "").strip().casefold().replace("_", " ")
    return {
        "healthy": "none",
        "normal": "none",
        "poisoned": "poison",
        "paralyzed": "paralysed",
    }.get(normalized, normalized)


def _item_cures_status(item: RecoveryItem, status: Any) -> bool:
    """Match live status names to the normalized recovery coverage."""

    normalized = _normalized_status(status)
    if normalized in {"", "none"}:
        return False
    coverage = {_normalized_status(value) for value in item.status_healed}
    return normalized in coverage or "all" in coverage


def _battle_party_member(battler: Any, party: Any) -> Any | None:
    """Return the party record for a battler's exact ROM party slot."""

    if battler is None or party is None:
        return None
    party_index = getattr(battler, "party_index", getattr(battler, "index", None))
    if not isinstance(party_index, int):
        return None
    try:
        member = party[party_index]
    except (IndexError, KeyError, TypeError):
        member = None
    if member is not None:
        member_index = getattr(member, "index", None)
        if member_index is None or member_index == party_index:
            return member
    try:
        return next(
            member
            for member in party
            if getattr(member, "index", None) == party_index
        )
    except (StopIteration, TypeError):
        return None


def _battle_moves(battler: Any, party: Any) -> tuple[Any, ...]:
    """Use the live battle moves, falling back to the same party slot.

    Emerald can expose a populated HP buffer before it has copied the active
    Pokémon's moves into ``gBattleMons``.  An empty move buffer is therefore a
    transient observation at the action boundary, not evidence that the
    Pokémon has no legal action.
    """

    try:
        moves = tuple(move for move in (getattr(battler, "moves", ()) or ()) if move is not None)
    except (AttributeError, RuntimeError, TypeError, ValueError, IndexError):
        moves = ()
    member = _battle_party_member(battler, party)
    try:
        fallback_moves = tuple(move for move in (getattr(member, "moves", ()) or ()) if move is not None)
    except (AttributeError, RuntimeError, TypeError, ValueError, IndexError):
        fallback_moves = ()
    if not moves:
        return fallback_moves

    # A partially initialized battle buffer may expose move identities while
    # reporting zero PP for every move. Prefer a same-slot party snapshot when
    # it still has usable PP; a real exhausted moveset remains exhausted.
    battle_has_pp = any(int(getattr(move, "pp", 1) or 0) > 0 for move in moves)
    fallback_has_pp = any(int(getattr(move, "pp", 0) or 0) > 0 for move in fallback_moves)
    if not battle_has_pp and fallback_has_pp:
        diagnostic_print(
            lambda: (
                "BATTLE_MOVES_NORMALIZATION: "
                f"party_index={getattr(battler, 'party_index', None)!r} "
                f"battle_moves={tuple(getattr(getattr(move, 'move', move), 'name', None) for move in moves)!r} "
                f"party_moves={tuple(getattr(getattr(move, 'move', move), 'name', None) for move in fallback_moves)!r}"
            ),
            trace=True,
            prefix="BATTLE_MOVES_NORMALIZATION:",
        )
        return fallback_moves
    return moves


def _battle_hp_facts(battler: Any, party: Any) -> tuple[int, int]:
    """Read HP from the battle buffer, with a matching party snapshot fallback.

    The action callback can become ready while ``gBattleMons`` is still being
    populated.  In that short window the battle battler may report zero HP
    even though the corresponding party buffer already contains the stable
    living Pokémon.  A zero battle value is never replaced by an arbitrary
    party member: only the same ROM party index may supply the fallback.
    """

    try:
        current_hp = max(0, int(getattr(battler, "current_hp", 0) or 0))
        max_hp = max(0, int(getattr(battler, "total_hp", 0) or 0))
    except (AttributeError, RuntimeError, TypeError, ValueError, IndexError):
        current_hp = 0
        max_hp = 0

    if current_hp > 0 and max_hp > 0:
        return current_hp, max_hp

    try:
        party_index = getattr(battler, "party_index", None)
        fallback = _battle_party_member(battler, party)
        fallback_current = max(0, int(getattr(fallback, "current_hp", 0) or 0))
        fallback_max = max(0, int(getattr(fallback, "total_hp", 0) or 0))
        if fallback_current > 0 and fallback_max > 0:
            diagnostic_print(
                lambda: (
                    "BATTLE_HP_NORMALIZATION: "
                    f"party_index={party_index!r} battle_hp={current_hp!r}/{max_hp!r} "
                    f"party_hp={fallback_current!r}/{fallback_max!r}"
                ),
                trace=True,
                prefix="BATTLE_HP_NORMALIZATION:",
            )
            return fallback_current, fallback_max
    except (AttributeError, RuntimeError, TypeError, ValueError, IndexError, KeyError):
        pass

    return current_hp, max_hp


def _knowledge_hp_facts(
    knowledge: BattleKnowledge | None,
    battler: Any,
    *,
    side: str,
) -> tuple[int, int] | None:
    """Recover same-battle HP when the current ROM buffer is incomplete.

    ``BattleState`` is read at the action callback boundary, while the
    battler buffers can still contain an all-zero structure for one frame.
    The observation recorder has already read the same battle on preceding
    frames. Use only the most recent matching observation, and never walk
    past an explicit faint, so an old species or a defeated battler cannot be
    revived by this fallback.
    """

    if knowledge is None or battler is None:
        return None
    try:
        party_index = getattr(battler, "party_index", getattr(battler, "index", None))
        species = getattr(getattr(battler, "species", None), "name", None)
        observations = getattr(knowledge, "observations", ())
        for observation in reversed(observations):
            candidates = getattr(observation, f"{side}_active", ())
            for candidate in candidates:
                candidate_index = getattr(candidate, "party_index", None)
                candidate_species = getattr(candidate, "species", None)
                if party_index is not None and candidate_index != party_index:
                    continue
                if species is not None and candidate_species is not None and candidate_species != species:
                    continue
                current = getattr(candidate, "current_hp", None)
                maximum = getattr(candidate, "max_hp", None)
                # A zero/zero pair is the incomplete ROM shape this helper is
                # intended to repair. An explicit zero with a known maximum
                # is a real faint and must remain authoritative.
                if current == 0 and maximum == 0:
                    continue
                if current is None or maximum is None:
                    continue
                current = int(current)
                maximum = int(maximum)
                if current > 0 and maximum > 0:
                    return current, maximum
                return None
    except (AttributeError, RuntimeError, TypeError, ValueError, IndexError, KeyError):
        return None
    return None


def _recovery_item(item: Any, quantity: int) -> RecoveryItem:
    """Normalize Emerald's item table into HP/status effects.

    The ROM item table exposes a numeric parameter, while status cures are
    encoded by item identity. Keep that translation at the live adapter so
    the planner receives one consistent RecoveryItem contract.
    """

    name = str(getattr(item, "name", "")).casefold()
    status_map = {
        "antidote": frozenset({"poison", "badly poisoned"}),
        "burn heal": frozenset({"burned"}),
        "ice heal": frozenset({"frozen"}),
        "awakening": frozenset({"asleep"}),
        "paralyze heal": frozenset({"paralysed"}),
    }
    full_restore = name == "full restore"
    full_status = name in {"full heal", "full restore"}
    full_hp = full_restore or name == "max potion"
    return RecoveryItem(
        item=item,
        quantity=quantity,
        heal_amount=0 if full_hp else int(getattr(item, "parameter", 0) or 0),
        status_healed=frozenset({"ALL"}) if full_status else status_map.get(name, frozenset()),
        full_heal=full_hp,
        utility_tier={"full restore": 5, "max potion": 4, "hyper potion": 3, "super potion": 2, "potion": 1}.get(
            name, 0
        ),
    )


def _capture_ball_multiplier(battle_state: Any, ball: Any) -> float:
    """Return the Emerald catch-rate multiplier for one normalized ball."""

    from modules.map import get_map_data_for_current_position
    from modules.pokedex import get_pokedex
    from modules.pokemon import get_type_by_name

    opponent = battle_state.opponent.active_battler
    multiplier = 1.0
    match getattr(ball, "index", None):
        case 1:  # Master Ball; retaining the legacy negative marker avoids auto-selection.
            multiplier = -1.0
        case 2:
            multiplier = 2.0
        case 3 | 5:
            multiplier = 1.5
        case 6:
            if opponent.species.has_type(get_type_by_name("Water")) or opponent.species.has_type(get_type_by_name("Bug")):
                multiplier = 3.0
        case 7:
            if get_map_data_for_current_position().map_type == "Underwater":
                multiplier = 3.5
        case 8:
            if opponent.level < 40:
                multiplier = max(1.0, (40 - opponent.level) / 10)
        case 9:
            if opponent.species in get_pokedex().owned_species:
                multiplier = 3.0
        case 10:
            multiplier = min(4.0, (10 + battle_state.current_turn) / 10)
    return multiplier


def plan_battle_state(
    battle_state: Any,
    knowledge: BattleKnowledge | None = None,
    *,
    incoming_opponent: Any | None = None,
    free_switch: bool = False,
) -> PlannerDecision:
    """Build planner facts from the live battle APIs, then make one decision.

    Opponent moves are read only from human-visible knowledge.  In particular,
    this function never consults ``BattlePokemon.moves`` for the opponent.
    """
    from modules.battle_strategies._util import BattleStrategyUtil
    from modules.items import ItemBattleUse, get_item_bag
    from modules.pokemon import StatusCondition, get_move_by_name
    from modules.pokemon_party import get_opponent_party, get_party

    active = battle_state.own_side.active_battler
    # A Switch prompt is raised after one trainer battler faints but before
    # the next battler is installed. Let the same normalized planner evaluate
    # that incoming roster member instead of making the replacement policy
    # reconstruct a second battle model.
    opponent = incoming_opponent or battle_state.opponent.active_battler
    player_party = get_party()
    active_hp, active_max_hp = _battle_hp_facts(active, player_party)
    opponent_party = None
    try:
        opponent_current_hp = int(getattr(opponent, "current_hp", 0) or 0)
        opponent_total_value = getattr(opponent, "total_hp", None)
        opponent_total_hp = int(opponent_total_value or 0) if opponent_total_value is not None else None
        if opponent_current_hp <= 0 or (opponent_total_hp is not None and opponent_total_hp <= 0):
            opponent_party = get_opponent_party()
    except (AttributeError, RuntimeError, TypeError, ValueError, IndexError, KeyError):
        opponent_party = None
    opponent_hp, opponent_max_hp = _battle_hp_facts(opponent, opponent_party)
    for label, battler, current_facts, side in (
        ("active", active, (active_hp, active_max_hp), "player"),
        ("opponent", opponent, (opponent_hp, opponent_max_hp), "opponent"),
    ):
        if current_facts[0] > 0 and current_facts[1] > 0:
            continue
        observed_facts = _knowledge_hp_facts(knowledge, battler, side=side)
        if observed_facts is None:
            continue
        if label == "active":
            active_hp, active_max_hp = observed_facts
        else:
            opponent_hp, opponent_max_hp = observed_facts
        diagnostic_print(
            lambda label=label, current_facts=current_facts, observed_facts=observed_facts: (
                "BATTLE_KNOWLEDGE_HP_NORMALIZATION: "
                f"side={label!r} live_hp={current_facts[0]!r}/{current_facts[1]!r} "
                f"observed_hp={observed_facts[0]!r}/{observed_facts[1]!r}"
            ),
            trace=True,
            prefix="BATTLE_KNOWLEDGE_HP_NORMALIZATION:",
        )
    active_moves = _battle_moves(active, player_party)
    player_bide_turns, player_bide_damage = _bide_facts(active)
    opponent_bide_turns, opponent_bide_damage = _bide_facts(opponent)
    if opponent_bide_turns == 1 and getattr(opponent, "bide_damage_known", True) is False:
        # The release is next but its stored damage could not be read.  Zero
        # would make the release look harmless, so assume the worst case: a
        # release (twice the stored damage) that can faint the active Pokémon.
        opponent_bide_damage = max(opponent_bide_damage, (int(active_hp) + 1) // 2)
        print_always(
            lambda: (
                "BATTLE_BIDE_DAMAGE_UNKNOWN: opponent Bide release is next but gBideDmg is "
                f"unreadable; assuming stored_damage={opponent_bide_damage!r}"
            ),
        )
    if opponent_bide_turns > 0:
        print_always(
            lambda: (
                "BATTLE_BIDE_OBSERVED: "
                f"opponent={getattr(getattr(opponent, 'species', None), 'name', None)!r} "
                f"turns_remaining={opponent_bide_turns!r} stored_damage={opponent_bide_damage!r} "
                f"release_damage={opponent_bide_damage * 2!r} active_hp={active_hp!r}"
            ),
        )
    util = BattleStrategyUtil(battle_state)
    visible = knowledge_for_planner(knowledge)
    revealed = {name for _, name in visible.revealed_opponent_moves} if visible is not None else set()
    species_move_names: set[str] = (
        _wild_level_up_move_names(opponent)
        if not getattr(battle_state, "is_trainer_battle", False)
        else set()
    )
    response_names = set(opponent_response_move_names(battle_state, visible))
    known_trainer_moves = set()
    if getattr(battle_state, "is_trainer_battle", False):
        known_trainer_moves = set(known_moves_for_battle(battle_state, opponent))
        response_names.update(known_trainer_moves)
    # A trainer's revealed moves are evidence of possible responses, not an
    # exhaustive moveset. For a wild Pokémon, the level-up envelope is the
    # complete game-legal source unless an observation introduced a move that
    # is outside that envelope.
    response_facts_complete = bool(
        (
            not getattr(battle_state, "is_trainer_battle", False)
            and species_move_names
            and response_names <= species_move_names | revealed
        )
        or (getattr(battle_state, "is_trainer_battle", False) and known_trainer_moves)
    )

    def damage(attacker, defender, move_name):
        try:
            move = get_move_by_name(move_name)
            if _is_bide_move(move):
                return 0
            return util.calculate_move_damage_range(move, attacker, defender).max
        except (AttributeError, KeyError, RuntimeError, TypeError, ValueError):
            return None

    unsupported_opponent_moves = []
    opponent_moves = []
    for name in sorted(response_names):
        if not name:
            continue
        try:
            move = get_move_by_name(name)
            if not _mechanics_supported(move):
                unsupported_opponent_moves.append(name)
            if _is_bide_move(move):
                damage_min = damage_max = 0
                critical_damage_max = 0
            else:
                range_ = util.calculate_move_damage_range(move, opponent, active)
                damage_min = range_.min
                damage_max = range_.max
                try:
                    critical_damage_max = util.calculate_move_damage_range(
                        move,
                        opponent,
                        active,
                        is_critical_hit=True,
                    ).max
                except (AttributeError, RuntimeError, TypeError, ValueError):
                    critical_damage_max = None
            opponent_moves.append(
                PlannerOpponentMove(
                    name,
                    damage_min,
                    damage_max,
                    getattr(move, "accuracy", 1.0),
                    getattr(move, "priority", 0),
                    _mechanics_supported(move),
                    (
                        "observed"
                        if name in revealed
                        else "species level-up envelope"
                        if name in species_move_names
                        else "known"
                    ),
                    critical_damage_max,
                    1 / 16,
                    response_facts_complete,
                    getattr(move, "secondary_accuracy", 0.0),
                    getattr(move, "effect", None),
                    getattr(move, "effect", None) == "TELEPORT",
                    _damage_stat_for_move(move),
                )
            )
        except (AttributeError, KeyError, RuntimeError, TypeError, ValueError):
            continue
    response_facts_complete = (
        response_facts_complete
        and bool(response_names)
        and len(opponent_moves) == len(response_names)
        and not unsupported_opponent_moves
    )
    unsupported_opponent_moves = tuple(unsupported_opponent_moves)
    opponent_damage_max = max((candidate.damage_max for candidate in opponent_moves), default=None)
    moves = []
    for index, learned in enumerate(active_moves):
        if learned is None or learned.pp <= 0 or learned.move.name in context.config.battle.banned_moves:
            continue
        try:
            if active.disabled_move is learned.move or getattr(active.disabled_move, "name", None) == learned.move.name:
                continue
            if _is_bide_move(learned.move):
                damage_min = damage_max = 0
            else:
                range_ = util.calculate_move_damage_range(learned.move, active, opponent)
                damage_min = range_.min
                damage_max = range_.max
            moves.append(
                PlannerMove(
                    index,
                    learned.move.name,
                    damage_min,
                    damage_max,
                    _mechanics_supported(learned.move),
                    getattr(learned.move, "accuracy", 1.0),
                    getattr(learned.move, "priority", 0),
                    getattr(learned.move, "effect", None),
                    damage_stat=_damage_stat_for_move(learned.move),
                    turns_required=2 if getattr(learned.move, "effect", None) == "SEMI_INVULNERABLE" else 1,
                    secondary_chance=getattr(learned.move, "secondary_accuracy", 0.0),
                    pp=learned.pp,
                )
            )
        except (AttributeError, RuntimeError, TypeError, ValueError):
            continue

    switches = []
    for party_index, pokemon in enumerate(player_party):
        if party_index == active.party_index or pokemon.is_egg or pokemon.current_hp <= 0:
            continue
        incoming = [
            value
            for opponent_move in opponent_moves
            if (value := damage(opponent, pokemon, opponent_move.name)) is not None
        ]
        incoming_critical = []
        for opponent_move in opponent_moves:
            try:
                move = get_move_by_name(opponent_move.name)
                if _is_bide_move(move):
                    incoming_critical.append(0)
                else:
                    incoming_critical.append(
                        util.calculate_move_damage_range(
                            move,
                            opponent,
                            pokemon,
                            is_critical_hit=True,
                        ).max
                    )
            except (AttributeError, KeyError, RuntimeError, TypeError, ValueError):
                continue
        offensive = []
        replacement_moves = []
        for learned in pokemon.moves:
            if (
                learned is not None
                and learned.pp > 0
                and learned.move.base_power > 0
                and learned.move.name not in context.config.battle.banned_moves
            ):
                value = damage(pokemon, opponent, learned.move.name)
                if value is not None:
                    offensive.append(value)
                    try:
                        if _is_bide_move(learned.move):
                            replacement_min = replacement_max = 0
                        else:
                            replacement_range = util.calculate_move_damage_range(learned.move, pokemon, opponent)
                            replacement_min = replacement_range.min
                            replacement_max = replacement_range.max
                        replacement_moves.append(
                            PlannerMove(
                                len(replacement_moves),
                                learned.move.name,
                                replacement_min,
                                replacement_max,
                                _mechanics_supported(learned.move),
                                getattr(learned.move, "accuracy", 1.0),
                                getattr(learned.move, "priority", 0),
                                getattr(learned.move, "effect", None),
                                damage_stat=_damage_stat_for_move(learned.move),
                                turns_required=2
                                if getattr(learned.move, "effect", None) == "SEMI_INVULNERABLE"
                                else 1,
                                secondary_chance=getattr(learned.move, "secondary_accuracy", 0.0),
                                pp=learned.pp,
                            )
                        )
                    except (AttributeError, RuntimeError, TypeError, ValueError):
                        pass
        switches.append(
            PlannerSwitch(
                party_index,
                pokemon.name,
                pokemon.current_hp,
                pokemon.total_hp,
                max(incoming) if incoming else None,
                max(offensive, default=0),
                max(incoming_critical) if incoming_critical else None,
                _status_value(getattr(pokemon, "status_condition", None)),
                tuple(replacement_moves),
                getattr(getattr(pokemon, "stats", None), "speed", None),
                escape_guaranteed=(
                    getattr(getattr(pokemon, "stats", None), "speed", None) is not None
                    and getattr(getattr(opponent, "stats", None), "speed", None) is not None
                    and getattr(getattr(pokemon, "stats", None), "speed", None)
                    > getattr(getattr(opponent, "stats", None), "speed", None)
                ),
            )
        )

    capture_target = bool(getattr(battle_state, "nuzlocke_capture_target", False)) and not battle_state.is_trainer_battle
    items = []
    try:
        for slot in get_item_bag().items:
            if slot.item.battle_use is ItemBattleUse.Healing:
                items.append(_recovery_item(slot.item, slot.quantity))
        # Poké Balls are objective facts.  Keeping them out of ordinary
        # battle contexts prevents unrelated planner diagnostics and search
        # state from treating inventory as a battle resource.
        if capture_target:
            for slot in get_item_bag().poke_balls:
                items.append(RecoveryItem(slot.item, slot.quantity, 0, True))
    except (AttributeError, RuntimeError, TypeError, ValueError):
        pass

    capture_ball_count = sum(
        max(0, int(item.quantity))
        for item in items
        if item.is_capture_ball
    )
    capture_available = capture_target and capture_ball_count > 0

    capture_probability_for_state = getattr(context, "battle_capture_probability_for_state", None)
    capture_success_probability = getattr(context, "battle_capture_success_probability", None)
    if capture_target and capture_available and not callable(capture_probability_for_state):
        balls_by_name = {
            getattr(item.item, "name", ""): item.item
            for item in items
            if item.is_capture_ball and item.quantity > 0
        }
        status_by_value = {condition.value: condition for condition in StatusCondition}

        def capture_probability_for_state(search_state, search_item) -> float:
            ball = balls_by_name.get(getattr(search_item, "name", ""))
            if ball is None:
                return 0.0
            status = status_by_value.get(
                getattr(search_state, "opponent_status", None),
                StatusCondition.Healthy,
            )
            return util.calculate_catch_success_chance(
                battle_state,
                _capture_ball_multiplier(battle_state, ball),
                current_hp=getattr(search_state, "opponent_hp", opponent_hp),
                status=status,
            )

        current_status = _status_value(getattr(opponent, "status_permanent", None))
        capture_success_probability = max(
            (
                util.calculate_catch_success_chance(
                    battle_state,
                    _capture_ball_multiplier(battle_state, ball),
                    current_hp=opponent_hp,
                    status=status_by_value.get(current_status, StatusCondition.Healthy),
                )
                for ball in balls_by_name.values()
            ),
            default=0.0,
        )

    escape = 0.0
    escape_attempts = None
    if not battle_state.is_trainer_battle:
        try:
            escape = util.get_escape_chance()
            get_escape_attempts = getattr(util, "get_escape_attempts", None)
            if callable(get_escape_attempts):
                escape_attempts = get_escape_attempts()
        except (AttributeError, RuntimeError, TypeError, ValueError):
            escape = 0.0
            escape_attempts = None
    escape_confidence = (
        PlannerConfidence.HIGH
        if escape == 1
        else PlannerConfidence.MEDIUM if escape >= 0.5 else PlannerConfidence.INSUFFICIENT
    )
    switch_streak, switch_cycle_pressure = _observed_switch_pressure(visible)
    planner_context = PlannerContext(
        active.species.name,
        active_hp,
        active_max_hp,
        opponent.species.name,
        opponent_hp,
        tuple(moves),
        tuple(switches),
        tuple(items),
        opponent_damage_max,
        tuple(opponent_moves),
        getattr(getattr(active, "stats", None), "speed", None),
        getattr(getattr(opponent, "stats", None), "speed", None),
        not battle_state.is_trainer_battle,
        battle_state.is_trainer_battle,
        util.can_switch() or free_switch,
        escape > 0,
        escape_confidence,
        opponent_max_hp,
        capture_target,
        capture_available,
        bool(getattr(battle_state, "capture_attempted", False)),
        max(
            (
                candidate.critical_damage_max
                if candidate.critical_damage_max is not None
                else candidate.damage_max
                for candidate in opponent_moves
            ),
            default=None,
        ),
        active_party_index=active.party_index,
        opponent_response_facts_complete=response_facts_complete,
        active_status=_status_value(getattr(active, "status_permanent", None)),
        battle_objective=normalized_battle_objective(
            battle_state,
            capture_target=capture_target,
            capture_available=capture_available,
        ),
        capture_success_probability=capture_success_probability if capture_target else None,
        capture_probability_for_state=capture_probability_for_state if capture_target else None,
        experience_reward=int(getattr(context, "battle_experience_reward", 0) or 0),
        experience_values=tuple(getattr(context, "battle_experience_values", ()) or ()),
        active_stat_stages=_planner_stat_stages(active),
        opponent_stat_stages=_planner_stat_stages(opponent),
        capture_ball_count=capture_ball_count if capture_target else 0,
        escape_probability=escape,
        escape_attempts=escape_attempts,
        escape_guaranteed=escape == 1.0,
        campaign_owned=getattr(context, "bot_mode", None) == "Campaign Progression",
        switch_history=(
            tuple(
                index
                for event in (getattr(visible, "switch_history", ()) if visible is not None else ())
                if getattr(event, "side", None) == "player"
                for index in (getattr(event, "before_party_index", None), getattr(event, "after_party_index", None))
                if index is not None
            )[-4:]
        ),
        free_switch=free_switch,
        switch_streak=switch_streak,
        switch_cycle_pressure=switch_cycle_pressure,
        unsupported_opponent_moves=unsupported_opponent_moves,
        unsupported_response_ceiling=active_max_hp if unsupported_opponent_moves else 0,
        player_bide_turns=player_bide_turns,
        player_bide_damage=player_bide_damage,
        opponent_bide_turns=opponent_bide_turns,
        opponent_bide_damage=opponent_bide_damage,
    )
    _log_battle_kind(battle_state, knowledge, planner_context)
    objective_facts = (
        f"capture_available={planner_context.capture_available} "
        f"capture_ball_count={planner_context.capture_ball_count}"
        if planner_context.capture_target
        else "capture_diagnostics=omitted"
    )
    if capture_target and not capture_available:
        diagnostic_print(
            lambda: (
                "BATTLE_OBJECTIVE_TRANSITION: "
                "from='capture' to='defeat' reason='no usable Poké Balls' "
                f"balls_remaining={capture_ball_count}"
            ),
            trace=True,
        )
    diagnostic_print(
            lambda: (
                "BATTLE_PLANNER_FACTS: "
                f"human_knowledge_policy={getattr(visible, 'policy', None)} "
                f"known_trainer_moves={sorted(known_trainer_moves)!r} "
                f"opponent_moves={[move.name for move in planner_context.opponent_moves]!r} "
                f"response_facts_complete={planner_context.opponent_response_facts_complete!r} "
                f"opponent_damage_max={planner_context.opponent_damage_max!r} "
            f"candidate_moves={[move.name for move in planner_context.moves]!r} "
            f"active_hp={planner_context.active_hp}/{planner_context.active_max_hp} "
            f"opponent_hp={planner_context.opponent_hp} "
            f"active_stat_stages={dict(planner_context.active_stat_stages)!r} "
                f"opponent_stat_stages={dict(planner_context.opponent_stat_stages)!r} "
                f"battle_objective={planner_context.battle_objective!r} "
                f"nuzlocke_capture_target={planner_context.capture_target} "
                f"campaign_owned={planner_context.campaign_owned!r} "
                f"switch_streak={planner_context.switch_streak!r} "
                f"switch_cycle_pressure={planner_context.switch_cycle_pressure!r} "
                f"unsupported_opponent_moves={planner_context.unsupported_opponent_moves!r} "
                f"unsupported_response_ceiling={planner_context.unsupported_response_ceiling!r} "
                f"player_bide=({planner_context.player_bide_turns!r}, {planner_context.player_bide_damage!r}) "
                f"opponent_bide=({planner_context.opponent_bide_turns!r}, {planner_context.opponent_bide_damage!r}) "
                f"{objective_facts}"
        ),
        trace=True,
    )
    decision = BattlePlanner().plan(planner_context)
    if decision.stochastic_result is None:
        diagnostic_print(
            lambda: (
                "BATTLE_PLANNER_EMPTY_ROOT: "
                f"active={planner_context.active_name!r} "
                f"active_hp={planner_context.active_hp!r}/{planner_context.active_max_hp!r} "
                f"opponent={planner_context.opponent_name!r} "
                f"opponent_hp={planner_context.opponent_hp!r}/{planner_context.opponent_max_hp!r} "
                f"candidate_moves={tuple((move.index, move.name, move.damage_min, move.damage_max) for move in planner_context.moves)!r} "
                f"items={tuple((getattr(getattr(item, 'item', None), 'name', None), item.quantity, item.is_capture_ball) for item in planner_context.items)!r} "
                f"capture_target={planner_context.capture_target!r} "
                f"capture_available={planner_context.capture_available!r} "
                f"capture_ball_count={planner_context.capture_ball_count!r} "
                f"response_facts_complete={planner_context.opponent_response_facts_complete!r} "
                f"opponent_moves={tuple(move.name for move in planner_context.opponent_moves)!r} "
                f"opponent_damage_max={planner_context.opponent_damage_max!r} "
                f"battle_objective={planner_context.battle_objective!r}"
            ),
            trace=False,
            prefix="BATTLE_PLANNER_EMPTY_ROOT",
        )
    diagnostic_print(lambda: format_planner_decision_diagnostic(decision, planner_context), trace=True)
    return decision


def _mechanics_supported(move: Any) -> bool:
    """Effects not represented by the bounded evaluator must stay conservative."""
    # Bide is modeled explicitly by expectiminimax: it locks the user, stores
    # actual direct HP damage, and releases double that damage. Keep the
    # remaining reflected/dynamic-damage mechanics conservative until they
    # have equivalent stateful models.
    unsupported_names = {"Counter", "Mirror Coat", "Endeavor"}
    unsupported_effects = {"COUNTER", "MIRROR_COAT", "ENDEAVOR", "MULTI_HIT", "TWO_TURNS"}
    return (
        getattr(move, "name", None) not in unsupported_names
        and getattr(move, "effect", None) not in unsupported_effects
    )


def _is_bide_move(move: Any) -> bool:
    return str(getattr(move, "name", "") or "").strip().casefold() == "bide" or str(
        getattr(move, "effect", "") or ""
    ).strip().upper() == "BIDE"


def _damage_stat_for_move(move: Any) -> str:
    """Return the Gen III attacking stat used by a move."""

    return (
        "special_attack"
        if getattr(getattr(move, "type", None), "kind", None) == "Special"
        else "attack"
    )


def _planner_stat_stages(pokemon: Any) -> tuple[tuple[str, int], ...]:
    """Return normalized live stat stages for the planner boundary."""

    try:
        modifiers = pokemon.stats_modifiers
        fields = (
            ("attack", "attack"),
            ("defence", "defense"),
            ("speed", "speed"),
            ("special_attack", "special_attack"),
            ("special_defence", "special_defense"),
            ("accuracy", "accuracy"),
            ("evasion", "evasion"),
        )
        stages = []
        for source, normalized in fields:
            try:
                value = int(getattr(modifiers, source))
            except (AttributeError, IndexError, KeyError, RuntimeError, TypeError, ValueError):
                continue
            stages.append((normalized, max(-6, min(6, value))))
        return tuple(stages)
    except (AttributeError, IndexError, KeyError, RuntimeError, TypeError, ValueError):
        return ()
