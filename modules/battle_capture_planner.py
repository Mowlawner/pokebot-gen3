"""Dedicated, bounded decision procedure for wild captures.

A wild capture is not an adversarial game.  The wild Pokémon is not trying to
stop the capture; the question is how to maximize the chance of catching it
without losing the Pokémon in play, given finitely many balls.  The generic
expectiminimax search treats it as a minimax tree and then scores leaves with an
unbounded rollout; this module answers the capture question directly and in
bounded time.

Each turn it compares a small family of plans: throw now (``k = 0``), or use one
damaging move ``k`` times (``k = 1..max_weaken_turns``) and then throw balls
until none remain.  A plan is evaluated exactly under a discrete outcome model
(accuracy, damage rolls, critical hits, the target's HP, the opponent's
strongest move, and the catch probability at each HP).  Re-planning every turn
from the live state recovers closed-loop behavior from these open-loop plans.

Decision rule, in order:

1. Only consider plans whose probability of losing the Pokémon in play is at
   most ``max_active_faint_probability``.
2. Among those, maximize the probability of capture.  This already weighs the
   risk of knocking the target out against the risk of running out of balls.
3. Among plans within ``tie_epsilon`` of the best capture probability, prefer
   the lower faint probability, then fewer expected balls, then fewer turns.

If the lead has no plan within that bound, a switch to another party member is
considered, but only to a member who can survive two worst-case opponent attacks
(a critical hit counts as an attack), only when switching is legal, and only
once per stretch of battle so it cannot ping-pong.  If still no plan is within
the bound, the least risky plan is chosen (lowest faint probability, then
highest capture probability) and is explicitly marked as risk accepted rather
than safe.  The facts needed to model the battle being unavailable still returns
``None``, and the caller falls back to the generic search.  It never invents
safety it cannot establish.

Known approximations (all documented because they affect probabilities):

- Damage rolls are three evenly spaced values between the planner's min and max.
- The opponent is modeled as always using its highest-damage known move, which
  is conservative for the faint bound.
- Status-inflicting and stat-changing moves are not used as capture aids.
- At most ``max_throws`` throws are modeled; further balls are ignored.
- In a branch that ends in a target KO, the lead can only faint during the
  opponent actions before the KO; the two outcomes are kept mutually exclusive.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from types import SimpleNamespace
from typing import Any

from modules.battle_model import DamageOutcome, move_outcomes

# Moves that only deal ordinary (possibly secondary-effect) damage and are safe
# to model as a plain damage roll.
_PLAIN_DAMAGE_EFFECTS = frozenset({"HIT", "FALSE_SWIPE", "ABSORB"})


@dataclass(frozen=True, slots=True)
class CaptureConfig:
    max_weaken_turns: int = 4
    max_throws: int = 12
    max_active_faint_probability: float = 0.03
    tie_epsilon: float = 0.005
    damage_points: int = 3


@dataclass(frozen=True, slots=True)
class CapturePlan:
    """One evaluated plan: ``weaken_turns`` uses of a move, then balls."""

    move_index: int | None
    move_name: str | None
    weaken_turns: int
    capture_probability: float
    active_faint_probability: float
    target_ko_probability: float
    expected_balls: float
    switch_party_index: int | None = None
    switch_name: str | None = None

    @property
    def first_action(self) -> str:
        if self.switch_party_index is not None:
            return f"switch to {self.switch_name}"
        return "throw" if self.weaken_turns == 0 else f"use {self.move_name}"

    def summary(self) -> str:
        label = "throw ball now" if self.weaken_turns == 0 else f"{self.move_name} x{self.weaken_turns} then throw"
        if self.switch_party_index is not None:
            label = f"switch to {self.switch_name}, then " + (
                "throw" if self.weaken_turns == 0 else f"{self.move_name} x{self.weaken_turns} then throw"
            )
        return (
            f"{label}: capture={self.capture_probability:.1%} "
            f"active_faint={self.active_faint_probability:.1%} "
            f"target_ko={self.target_ko_probability:.1%} "
            f"expected_balls={self.expected_balls:.1f}"
        )


@dataclass(frozen=True, slots=True)
class CaptureChoice:
    plan: CapturePlan
    ball_item: Any
    alternatives: tuple[CapturePlan, ...] = field(default_factory=tuple)
    # True when no plan met the faint bound and this is the least risky one.
    risk_accepted: bool = False


@dataclass(frozen=True, slots=True)
class CaptureSolverResult:
    """Minimal result for a risk-accepted decision.

    The campaign acceptance gate lets a decision that is not proven safe proceed
    only when its result declares itself an autonomous root.  This object is that
    declaration; it never claims the line is safe.
    """

    active_faint_probability: float
    capture_probability: float
    objective: str = "capture"
    complete: bool = True
    projection_safe: bool = False
    is_safe: bool = False
    objective_failed: bool = False
    is_autonomous_root: bool = True
    category: Any = None


def _is_plain_damaging_move(move: Any) -> bool:
    if not getattr(move, "mechanics_supported", True):
        return False
    if int(getattr(move, "damage_max", 0) or 0) <= 0:
        return False
    if int(getattr(move, "turns_required", 1) or 1) != 1:
        return False
    pp = getattr(move, "pp", None)
    if pp is not None and pp <= 0:
        return False
    effect = getattr(move, "effect", None)
    if effect in (None, ""):
        return True
    return effect in _PLAIN_DAMAGE_EFFECTS or str(effect).endswith("_HIT")


def _opponent_turn_outcomes(context: Any, config: CaptureConfig) -> tuple[DamageOutcome, ...] | None:
    """Return the per-turn damage distribution of the opponent's strongest move."""

    moves = [
        move
        for move in getattr(context, "opponent_moves", ())
        if getattr(move, "mechanics_supported", True)
    ]
    if moves:
        strongest = max(
            moves,
            key=lambda move: (
                int(getattr(move, "damage_max", 0) or 0),
                float(getattr(move, "accuracy", 1.0) or 0.0),
            ),
        )
        critical = getattr(strongest, "critical_damage_max", None)
        return move_outcomes(
            int(getattr(strongest, "damage_min", 0) or 0),
            int(getattr(strongest, "damage_max", 0) or 0),
            float(getattr(strongest, "accuracy", 1.0)),
            critical_chance=float(getattr(strongest, "critical_hit_chance", 1 / 16)),
            critical_damage=critical,
            points=config.damage_points,
        )
    ceiling = getattr(context, "opponent_damage_max", None)
    if ceiling is not None:
        return (DamageOutcome(1.0, int(ceiling)),)
    return None


def _faint_table(active_hp: int, damage: tuple[DamageOutcome, ...], turns: int) -> list[float]:
    """``table[n]`` is the probability the active Pokémon has fainted within ``n`` opponent actions."""

    alive: dict[int, float] = {int(active_hp): 1.0}
    table = [0.0]
    for _ in range(turns):
        next_alive: dict[int, float] = {}
        for hp, probability in alive.items():
            for outcome in damage:
                remaining = hp - outcome.damage
                if remaining > 0:
                    next_alive[remaining] = next_alive.get(remaining, 0.0) + probability * outcome.probability
        alive = next_alive
        table.append(max(0.0, min(1.0, 1.0 - sum(alive.values()))))
    return table


def _catch_probability(context: Any, ball: Any, opponent_hp: int) -> float:
    """Probability that one throw of ``ball`` captures at ``opponent_hp``."""

    callback = getattr(context, "capture_probability_for_state", None)
    state = SimpleNamespace(opponent_hp=opponent_hp, opponent_status=getattr(context, "opponent_status", None))
    item = SimpleNamespace(name=getattr(getattr(ball, "item", None), "name", ""))
    return max(0.0, min(1.0, float(callback(state, item))))


def _throw_probabilities(context: Any, balls: list[Any], opponent_hp: int, max_throws: int) -> list[float]:
    """Per-throw capture probabilities, using the best remaining ball first."""

    sequence: list[float] = []
    ranked = sorted(
        ((_catch_probability(context, ball, opponent_hp), ball) for ball in balls),
        key=lambda pair: pair[0],
        reverse=True,
    )
    for probability, ball in ranked:
        sequence.extend([probability] * max(0, int(ball.quantity)))
        if len(sequence) >= max_throws:
            break
    return sequence[:max_throws]


def _evaluate_plan(
    context: Any,
    config: CaptureConfig,
    balls: list[Any],
    faint: list[float],
    move: Any | None,
    weaken_turns: int,
    pre_actions: int = 0,
) -> CapturePlan:
    """Evaluate one plan.  ``pre_actions`` counts opponent actions taken before it
    starts (one, when the plan begins by switching: the incoming Pokémon takes a hit)."""

    opponent_hp = int(context.opponent_hp)

    # Weakening phase: a distribution over the target's HP after ``weaken_turns``.
    distribution: dict[int, float] = {opponent_hp: 1.0}
    ko_by_turn: dict[int, float] = {}
    if move is not None:
        false_swipe = getattr(move, "effect", None) == "FALSE_SWIPE"
        for turn in range(1, weaken_turns + 1):
            next_distribution: dict[int, float] = {}
            for hp, probability in distribution.items():
                for outcome in move_outcomes(
                    int(move.damage_min),
                    int(move.damage_max),
                    float(move.accuracy),
                    points=config.damage_points,
                ):
                    damage = min(outcome.damage, hp - 1) if false_swipe else outcome.damage
                    remaining = hp - damage
                    weight = probability * outcome.probability
                    if remaining <= 0:
                        ko_by_turn[turn] = ko_by_turn.get(turn, 0.0) + weight
                    else:
                        next_distribution[remaining] = next_distribution.get(remaining, 0.0) + weight
            distribution = next_distribution

    capture = 0.0
    faint_total = 0.0
    expected_balls = 0.0
    for hp, probability in distribution.items():
        throws = _throw_probabilities(context, balls, hp, config.max_throws)
        fail_before = 1.0  # probability every earlier throw failed
        for index, throw_probability in enumerate(throws, start=1):
            actions_before_throw = pre_actions + weaken_turns + index - 1
            alive = 1.0 - faint[min(actions_before_throw, len(faint) - 1)]
            reach = fail_before * alive
            expected_balls += probability * reach
            capture += probability * reach * throw_probability
            fail_before *= 1.0 - throw_probability

        # Faint risk: the opponent acts after each weakening move and after
        # each failed throw; fainting only matters while the attempt is alive.
        survive_throws = 1.0
        previous = faint[min(pre_actions + weaken_turns, len(faint) - 1)]
        faint_total += probability * previous  # during the switch and weakening turns
        for index, throw_probability in enumerate(throws, start=1):
            survive_throws *= 1.0 - throw_probability
            actions = pre_actions + weaken_turns + index
            current = faint[min(actions, len(faint) - 1)]
            faint_total += probability * survive_throws * max(0.0, current - previous)
            previous = current

    # The outcomes are mutually exclusive: a knockout only counts when the
    # lead survived the opponent actions before it, otherwise the lead fainted
    # first.  Counting both would make the probabilities sum past one.
    target_ko = 0.0
    for turn, mass in ko_by_turn.items():
        fainted_first = faint[min(pre_actions + turn - 1, len(faint) - 1)]
        target_ko += mass * (1.0 - fainted_first)
        faint_total += mass * fainted_first

    return CapturePlan(
        move_index=None if move is None else int(move.index),
        move_name=None if move is None else str(move.name),
        weaken_turns=weaken_turns,
        capture_probability=min(1.0, capture),
        active_faint_probability=min(1.0, faint_total),
        target_ko_probability=min(1.0, target_ko),
        expected_balls=expected_balls,
    )


def _usable_balls(context: Any) -> list[Any]:
    return [
        item
        for item in getattr(context, "items", ())
        if getattr(item, "is_capture_ball", False) and int(getattr(item, "quantity", 0) or 0) > 0
    ]


def evaluate_capture_plans(context: Any, config: CaptureConfig = CaptureConfig()) -> tuple[list[CapturePlan], list[Any]] | None:
    """Evaluate every plan, or ``None`` when the battle cannot be modeled soundly."""

    if getattr(context, "battle_objective", None) != "capture":
        return None
    if not (getattr(context, "capture_target", False) and getattr(context, "capture_available", False)):
        return None
    if getattr(context, "is_trainer", False):
        return None
    if not callable(getattr(context, "capture_probability_for_state", None)):
        return None
    if int(getattr(context, "active_hp", 0) or 0) <= 0 or int(getattr(context, "opponent_hp", 0) or 0) <= 0:
        return None
    # Unknown or unsupported opponent behavior cannot support a safety claim.
    if not getattr(context, "opponent_response_facts_complete", True):
        return None
    if getattr(context, "unsupported_opponent_moves", ()):
        return None
    damage = _opponent_turn_outcomes(context, config)
    if damage is None:
        return None
    balls = _usable_balls(context)
    if not balls:
        return None

    faint = _faint_table(
        int(context.active_hp),
        damage,
        config.max_weaken_turns + config.max_throws + 1,
    )
    plans = [_evaluate_plan(context, config, balls, faint, None, 0)]
    for move in getattr(context, "moves", ()):
        if not _is_plain_damaging_move(move):
            continue
        for turns in range(1, config.max_weaken_turns + 1):
            plans.append(_evaluate_plan(context, config, balls, faint, move, turns))
    return plans, balls


def _worst_case_hit(context: Any, config: CaptureConfig) -> int:
    """Largest single opponent attack, counting a critical hit as an attack."""

    damage = _opponent_turn_outcomes(context, config)
    return max((outcome.damage for outcome in damage), default=0) if damage else 0


def _eligible_switches(context: Any, config: CaptureConfig) -> list[Any]:
    """Party members the planner may switch to for a capture.

    A candidate must be able to survive two worst-case opponent attacks, the
    switch must be legal, and no switch may already have happened in this
    stretch of battle (so the lead cannot ping-pong).
    """

    if not getattr(context, "can_switch", False):
        return []
    if int(getattr(context, "switch_streak", 0) or 0) > 0:
        return []
    recent = set(getattr(context, "switch_history", ()) or ())
    ceiling = _worst_case_hit(context, config)
    return [
        candidate
        for candidate in getattr(context, "switches", ())
        if int(getattr(candidate, "hp", 0) or 0) > 2 * ceiling
        and getattr(candidate, "party_index", None) not in recent
        and getattr(candidate, "party_index", None) != getattr(context, "active_party_index", None)
    ]


def describe_switch_candidates(context: Any, config: CaptureConfig = CaptureConfig()) -> tuple[str, ...]:
    """One line per bench member saying why it can or cannot be switched to.

    The decision rationale uses this so a log shows what the switch rule saw
    (HP against the worst-case hit) and not only the plan that was chosen.
    """

    members = [
        candidate
        for candidate in getattr(context, "switches", ())
        if getattr(candidate, "party_index", None) != getattr(context, "active_party_index", None)
    ]
    if not members:
        return ("no bench members",)
    if not getattr(context, "can_switch", False):
        return ("switching is not legal now",)
    if int(getattr(context, "switch_streak", 0) or 0) > 0:
        return ("a switch already happened in this stretch of battle",)
    ceiling = _worst_case_hit(context, config)
    recent = set(getattr(context, "switch_history", ()) or ())
    lines = []
    for candidate in members:
        hp = int(getattr(candidate, "hp", 0) or 0)
        label = f"{getattr(candidate, 'name', '?')} (slot {getattr(candidate, 'party_index', '?')}) HP {hp}"
        if getattr(candidate, "party_index", None) in recent:
            lines.append(f"{label}: rejected, was just switched out")
        elif hp > 2 * ceiling:
            lines.append(f"{label}: eligible, survives two worst-case hits (worst hit {ceiling})")
        else:
            lines.append(
                f"{label}: rejected, needs HP > {2 * ceiling} to survive two worst-case hits (worst hit {ceiling})"
            )
    return tuple(lines)


def evaluate_switch_plans(context: Any, config: CaptureConfig, balls: list[Any]) -> list[CapturePlan]:
    """Plans that begin by switching: the incoming Pokémon takes one opponent action first."""

    damage = _opponent_turn_outcomes(context, config)
    if damage is None:
        return []
    plans: list[CapturePlan] = []
    for candidate in _eligible_switches(context, config):
        faint = _faint_table(
            int(candidate.hp), damage, config.max_weaken_turns + config.max_throws + 2
        )
        options: list[tuple[Any | None, int]] = [(None, 0)]
        for move in getattr(candidate, "moves", ()):
            if _is_plain_damaging_move(move):
                options.extend((move, turns) for turns in range(1, config.max_weaken_turns + 1))
        for move, turns in options:
            plan = _evaluate_plan(context, config, balls, faint, move, turns, pre_actions=1)
            plans.append(
                replace(
                    plan,
                    switch_party_index=int(candidate.party_index),
                    switch_name=str(candidate.name),
                )
            )
    return plans


def _best(plans: list[CapturePlan], config: CaptureConfig) -> CapturePlan:
    best_capture = max(plan.capture_probability for plan in plans)
    contenders = [plan for plan in plans if best_capture - plan.capture_probability <= config.tie_epsilon]
    return min(
        contenders,
        key=lambda plan: (
            plan.active_faint_probability,
            plan.expected_balls,
            plan.weaken_turns,
            -plan.capture_probability,
        ),
    )


def choose_capture_plan(context: Any, config: CaptureConfig = CaptureConfig()) -> CaptureChoice | None:
    """Pick the plan to act on, or ``None`` to defer to the generic search.

    In order: the best lead plan within the faint bound; otherwise the best
    plan that starts by switching to a member who can take two hits; otherwise
    the least risky plan overall, marked as risk accepted.
    """

    evaluated = evaluate_capture_plans(context, config)
    if evaluated is None:
        return None
    lead_plans, balls = evaluated
    # The first action of a "throw" plan is the ball with the best odds now.
    ball = max(balls, key=lambda item: _catch_probability(context, item, int(context.opponent_hp)))

    def within_bound(plans: list[CapturePlan]) -> list[CapturePlan]:
        return [plan for plan in plans if plan.active_faint_probability <= config.max_active_faint_probability]

    def others_of(chosen: CapturePlan, plans: list[CapturePlan]) -> tuple[CapturePlan, ...]:
        return tuple(
            sorted(
                (plan for plan in plans if plan is not chosen),
                key=lambda plan: (-plan.capture_probability, plan.active_faint_probability),
            )[:4]
        )

    admissible = within_bound(lead_plans)
    if admissible:
        chosen = _best(admissible, config)
        return CaptureChoice(chosen, ball, others_of(chosen, lead_plans))

    switch_plans = evaluate_switch_plans(context, config, balls)
    admissible_switches = within_bound(switch_plans)
    everything = lead_plans + switch_plans
    if admissible_switches:
        chosen = _best(admissible_switches, config)
        return CaptureChoice(chosen, ball, others_of(chosen, everything))

    # Nothing meets the bound: take the least risky plan, and say so.
    chosen = min(
        everything,
        key=lambda plan: (
            plan.active_faint_probability,
            -plan.capture_probability,
            plan.expected_balls,
        ),
    )
    return CaptureChoice(chosen, ball, others_of(chosen, everything), risk_accepted=True)


__all__ = [
    "CaptureChoice",
    "CaptureConfig",
    "CapturePlan",
    "CaptureSolverResult",
    "choose_capture_plan",
    "evaluate_capture_plans",
    "describe_switch_candidates",
    "evaluate_switch_plans",
]
