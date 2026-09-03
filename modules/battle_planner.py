"""Conservative, explainable battle planning.

The planner deliberately consumes normalized, planner-facing facts.  The
adapter at the bottom is the only part that knows about the live battle
objects; this keeps strategic reasoning out of battle menu input code.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any

from modules.battle_observation import BattleKnowledge, KnowledgePolicy
from modules.console import diagnostic_print


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


@dataclass(frozen=True, slots=True)
class PlannerSwitch:
    party_index: int
    name: str
    hp: int
    max_hp: int
    incoming_damage_max: int | None
    offensive_damage_max: int = 0


@dataclass(frozen=True, slots=True)
class PlannerItem:
    item: Any
    quantity: int
    heal_amount: int
    is_capture_ball: bool = False


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
    capture_appropriate_hp: int | None = None
    capture_attempted: bool = False


class BattlePlanner:
    """Choose only actions whose safety can be explained from known facts."""

    _MAX_FORECAST_TURNS = 6
    _MIN_SEQUENCE_ACCURACY = 0.8

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
        if context.active_max_hp <= 0 or context.opponent_hp <= 0:
            return self._abort("Active or opponent HP is unavailable.")

        usable = [move for move in context.moves if self._is_candidate(move)]
        if context.capture_target and context.capture_available and not context.is_trainer:
            # A ball is an observation/action boundary, never proof of a
            # capture. The observer will later emit PokemonCaptured.
            if context.capture_available and (
                context.capture_attempted
                or context.capture_appropriate_hp is None
                or context.opponent_hp <= context.capture_appropriate_hp
            ):
                return PlannerDecision(
                    PlannerAction.UseItem,
                    self._capture_item(context),
                    PlannerConfidence.HIGH,
                    "Legal Nuzlocke capture target is in capture state; attempt capture before dealing more damage.",
                    PlannerSafety.SAFE,
                    classification=PlannerDecisionClass.SAFE,
                    risk=PlannerRisk.LOW,
                )

            # Filter lethal moves before the ordinary battle policy sees them.
            # Minimum damage is enough to prove a guaranteed KO; maximum damage
            # is used for meaningful KO risk.
            nonlethal = [move for move in usable if move.damage_max < context.opponent_hp]
            usable = nonlethal
        # A guaranteed KO is safe even when the opponent's unrevealed move is
        # unknown: the opponent gets no response after the KO.
        guaranteed = [
            move
            for move in usable
            if move.damage_min >= context.opponent_hp and move.accuracy >= 1 and move.mechanics_supported
        ]
        if guaranteed:
            move = max(guaranteed, key=lambda candidate: (candidate.damage_min, candidate.damage_max))
            confidence = PlannerConfidence.HIGH if move.mechanics_supported else PlannerConfidence.MEDIUM
            return PlannerDecision(
                PlannerAction.UseMove,
                move.index,
                confidence,
                f"{move.name}: {move.damage_min}-{move.damage_max} damage; {context.opponent_name} has "
                f"{context.opponent_hp} HP; guaranteed KO. {context.active_name} has "
                f"{context.active_hp}/{context.active_max_hp} HP. No multi-turn survival is required, "
                f"so the move is considered safe. {self._response_note(context)}",
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
                f"{rationale} {response_note}",
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
            ]
            if context.can_switch
            else []
        )
        responses_supported = all(move.mechanics_supported for move in context.opponent_moves)
        candidates = (
            self._search(context, usable)
            if (context.opponent_moves or context.opponent_damage_max is not None)
            and responses_supported
            and not safe_switches
            else []
        )
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
            rationale = (
                f"{line}: best modeled line ({'guaranteed win' if state['win'] else 'strongest available progress'});"
                f" projected incoming damage {state['incoming']}; {state['turns']} turn(s); "
                f"sequence accuracy {state['accuracy']:.0%}; active HP {context.active_hp}/{context.active_max_hp} HP; "
                f"projected HP {max(0, context.active_hp - state['incoming'])}/{context.active_max_hp}; "
                f"risk {risk.value.upper()}. {risk_note}{uncertainty}"
            )
            if state["win"] and all(move.damage_max > 0 for move in sequence):
                rationale += f" estimated {state['turns']} hits to KO; safe sequence established."
            if any(move.damage_max == 0 for move in sequence):
                rationale += " Supported stat/effect changes are included in the damage and exposure comparison."
            if not state["known_response"]:
                rationale += " Opponent response is unknown, so survival is not claimed as safe."
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
            replacement = max(
                safe_switches, key=lambda candidate: (candidate.offensive_damage_max, candidate.hp / candidate.max_hp)
            )
            return PlannerDecision(
                PlannerAction.SwitchPokemon,
                replacement.party_index,
                PlannerConfidence.MEDIUM,
                f"{context.active_name} is at {context.active_hp}/{context.active_max_hp} HP; "
                f"{replacement.name} is expected to survive the known response and has the best available matchup.",
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
                "this is the best available action, not a safety claim.",
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

        if context.capture_target and context.capture_available:
            return PlannerDecision(
                PlannerAction.UseItem,
                self._capture_item(context),
                PlannerConfidence.MEDIUM,
                "No nonlethal damaging action is available; preserve the legal capture target and attempt a ball.",
                PlannerSafety.SAFE,
                classification=PlannerDecisionClass.SAFE,
                risk=PlannerRisk.MODERATE,
            )

        healing = [item for item in context.items if item.quantity > 0 and item.heal_amount > 0]
        if healing and context.opponent_damage_max is not None:
            item = max(healing, key=lambda candidate: candidate.heal_amount)
            if min(context.active_max_hp, context.active_hp + item.heal_amount) > context.opponent_damage_max:
                return PlannerDecision(
                    PlannerAction.UseItem,
                    (item.item, 0),
                    PlannerConfidence.MEDIUM,
                    f"{item.item.name} materially improves survival: healing is expected to put "
                    f"{context.active_name} above the known maximum incoming damage.",
                    PlannerSafety.SAFE,
                    (
                        PlannerEvidence(
                            PlannerEvidenceKind.DERIVED, "healing places HP above known maximum incoming damage"
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
    def _capture_item(context: PlannerContext):
        balls = [item for item in context.items if item.quantity > 0 and item.is_capture_ball]
        return (max(balls, key=lambda item: item.quantity).item, 0) if balls else None

    @classmethod
    def _is_candidate(cls, move: PlannerMove) -> bool:
        return move.mechanics_supported and (
            move.damage_max > 0 or move.effect in cls._STAT_EFFECTS or move.stat_delta != 0
        )

    def _search(self, context: PlannerContext, moves: list[PlannerMove]):
        """Small deterministic minimax-like search over our bounded action lines.

        It uses minimum player damage and maximum visible incoming damage for safety.
        Fallback ranking is comparative, not a fabricated probability estimate.
        """
        opponent_moves = context.opponent_moves
        known_response = bool(opponent_moves or context.opponent_damage_max is not None)
        incoming_max = max((m.damage_max for m in opponent_moves), default=context.opponent_damage_max or 0)
        incoming_critical_max = max(
            (m.critical_damage_max if m.critical_damage_max is not None else m.damage_max for m in opponent_moves),
            default=context.opponent_damage_max or 0,
        )
        results = []

        def visit(sequence, hp, opponent_hp, stages, incoming, critical_incoming, accuracy, turns):
            if turns and opponent_hp <= 0:
                results.append(
                    (
                        self._score(
                            context, sequence, hp, incoming, critical_incoming, turns, accuracy, True, known_response
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
                        },
                    )
                )
                return
            if turns >= self._MAX_FORECAST_TURNS:
                return
            for move in moves:
                next_stages = dict(stages)
                move_accuracy = max(0.0, min(1.0, move.accuracy))
                next_accuracy = accuracy * move_accuracy
                damage = self._scaled_damage(move, next_stages)
                next_opponent_hp = max(0, opponent_hp - damage)
                if move.effect in self._STAT_EFFECTS:
                    side, stat, delta = self._STAT_EFFECTS[move.effect]
                    key = (side, stat)
                    next_stages[key] = max(-6, min(6, next_stages.get(key, 0) + delta))
                # Unknown responses remain unknown; a known response is conservatively worst-case.
                player_first = self._player_acts_first(context, move, opponent_moves)
                response = incoming_max if known_response and (not player_first or next_opponent_hp > 0) else 0
                critical_response = (
                    incoming_critical_max if known_response and (not player_first or next_opponent_hp > 0) else 0
                )
                next_hp = hp - response
                next_incoming = incoming + response
                next_critical_incoming = critical_incoming + critical_response
                next_sequence = sequence + (move,)
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
                    )

        visit((), context.active_hp, context.opponent_hp, {}, 0, 0, 1.0, 0)
        return results

    @staticmethod
    def _stage_multiplier(stage: int) -> float:
        return (2 + stage) / 2 if stage >= 0 else 2 / (2 - stage)

    def _scaled_damage(self, move: PlannerMove, stages: dict) -> int:
        if move.damage_min <= 0:
            return 0
        # damage_min is already produced by the shared type/STAB/stat model. Stat
        # effects only apply a bounded Gen III stage multiplier on top of it.
        attack_stat = move.stat_target or "attack"
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
                return "A modeled critical-hit ceiling can be lethal before the line completes."
            return "The modeled response can KO before the line completes."
        if risk is PlannerRisk.HIGH:
            if state.get("accuracy", 1.0) < BattlePlanner._MIN_SEQUENCE_ACCURACY:
                return "The required sequence accuracy is too uncertain for a safety claim."
            return "The line depends on surviving a meaningful opposing response."
        if risk is PlannerRisk.MODERATE:
            return "Accuracy or action order leaves bounded uncertainty."
        return "No modeled response reaches lethal damage before the win."

    @classmethod
    def _score(cls, context, sequence, hp, incoming, critical_incoming, turns, accuracy, win, known_response):
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
        risk_rank = {
            PlannerRisk.LOW: 3,
            PlannerRisk.MODERATE: 2,
            PlannerRisk.HIGH: 1,
            PlannerRisk.CRITICAL: 0,
        }[risk]
        survived = hp > 0
        return (
            int(survived),
            int(win),
            risk_rank,
            int(hp),
            -incoming,
            -turns,
            accuracy,
            sum(m.damage_max == 0 for m in sequence),
            -sum(m.index for m in sequence),
        )

    def _evaluate_attack(self, context: PlannerContext, move: PlannerMove):
        if not move.mechanics_supported or move.damage_min <= 0 or move.accuracy <= 0:
            return None
        hits = (context.opponent_hp + move.damage_min - 1) // move.damage_min
        if hits <= 0 or hits > self._MAX_FORECAST_TURNS:
            return None
        sequence_accuracy = move.accuracy**hits
        if sequence_accuracy < self._MIN_SEQUENCE_ACCURACY:
            return None
        opponent_moves = context.opponent_moves
        if not opponent_moves and context.opponent_damage_max is not None:
            opponent_moves = (PlannerOpponentMove("known response", 0, context.opponent_damage_max),)
        if hits > 1 and not opponent_moves:
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
        incoming_exchanges = max(0, hits - 1) if player_first else hits
        projected_incoming = incoming * incoming_exchanges
        projected_critical_incoming = critical_incoming * incoming_exchanges
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
            f"sequence accuracy {sequence_accuracy:.0%}; projected incoming damage {projected_incoming}; "
            f"{order_note}; critical-hit ceiling {projected_critical_incoming}; safe sequence established."
        )

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
    def _attack_failure_reason(context: PlannerContext, move: PlannerMove) -> str:
        if not move.mechanics_supported:
            return f"{move.name}: unsupported mechanics materially reduce confidence."
        hits = (context.opponent_hp + move.damage_min - 1) // move.damage_min if move.damage_min > 0 else 0
        if hits > BattlePlanner._MAX_FORECAST_TURNS:
            return f"{move.name}: requires at least {hits} minimum-damage hits, beyond the forecast horizon."
        if move.accuracy < 1 and move.accuracy ** max(1, hits) < BattlePlanner._MIN_SEQUENCE_ACCURACY:
            return f"{move.name}: accuracy is too uncertain for the estimated {hits}-hit sequence."
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
                f"projected {projected_incoming} incoming damage over {hits} hits; the multi-turn sequence is rejected."
            )
        if (opponent_moves or context.opponent_damage_max is not None) and context.active_hp <= critical_incoming:
            return (
                f"{move.name}: a modeled critical-hit ceiling of {critical_incoming} damage can be lethal "
                f"before the {hits}-hit line completes; the sequence is risky rather than safe."
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
        return "Opponent possibilities: " + "; ".join(descriptions) + "."

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
    unknown = not bool(known_moves or context.opponent_damage_max is not None)
    setup = bool(move and (move.damage_max == 0 or move.effect or move.stat_delta))
    rejected = decision.rejected_candidates[0] if decision.rejected_candidates else "none recorded"
    fallback = decision.classification is PlannerDecisionClass.BEST_AVAILABLE and not decision.is_safe_to_execute
    reason = decision.rationale
    if decision.classification is PlannerDecisionClass.BEST_AVAILABLE:
        reason = "No fully safe line established; selecting best available legal action. " + reason
    return (
        f"BATTLE_DECISION_DETAIL: {decision.classification.name}: {selected}; action={decision.action.value}; "
        f"confidence={decision.confidence.value}; player_hp={context.active_hp}/{context.active_max_hp}; "
        f"opponent_hp={context.opponent_hp}/{context.opponent_max_hp or 'unknown'}; damage={damage}; "
        f"hits_to_ko={hits if hits is not None else 'n/a'}; known_opponent_moves={known_moves!r}; "
        f"opponent_response_unknown={unknown}; risk={decision.risk.value}; setup_line={setup}; "
        f"fallback={fallback}; primary_reason={reason}; strongest_rejected_alternative={rejected}"
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


def plan_battle_state(battle_state: Any, knowledge: BattleKnowledge | None = None) -> PlannerDecision:
    """Build planner facts from the live battle APIs, then make one decision.

    Opponent moves are read only from human-visible knowledge.  In particular,
    this function never consults ``BattlePokemon.moves`` for the opponent.
    """
    from modules.battle_strategies._util import BattleStrategyUtil
    from modules.items import ItemBattleUse, get_item_bag
    from modules.pokemon import get_move_by_name
    from modules.pokemon_party import get_party

    active = battle_state.own_side.active_battler
    opponent = battle_state.opponent.active_battler
    util = BattleStrategyUtil(battle_state)
    visible = knowledge_for_planner(knowledge)
    revealed = set()
    known = set()
    observed = set()
    if visible is not None:
        revealed = {name for _, name in visible.revealed_opponent_moves}
        observed = revealed
        known = {name for _, name in visible.known_opponent_moves}

    def damage(attacker, defender, move_name):
        try:
            move = get_move_by_name(move_name)
            return util.calculate_move_damage_range(move, attacker, defender).max
        except (AttributeError, KeyError, RuntimeError, TypeError, ValueError):
            return None

    opponent_moves = []
    for name in sorted(revealed | known):
        if not name:
            continue
        try:
            move = get_move_by_name(name)
            range_ = util.calculate_move_damage_range(move, opponent, active)
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
                    range_.min,
                    range_.max,
                    getattr(move, "accuracy", 1.0),
                    getattr(move, "priority", 0),
                    _mechanics_supported(move),
                    "observed" if name in observed else "known",
                    critical_damage_max,
                )
            )
        except (AttributeError, KeyError, RuntimeError, TypeError, ValueError):
            continue
    opponent_damage_max = max((candidate.damage_max for candidate in opponent_moves), default=None)
    moves = []
    for index, learned in enumerate(active.moves):
        if learned is None or learned.pp <= 0:
            continue
        try:
            if active.disabled_move is learned.move:
                continue
            range_ = util.calculate_move_damage_range(learned.move, active, opponent)
            moves.append(
                PlannerMove(
                    index,
                    learned.move.name,
                    range_.min,
                    range_.max,
                    _mechanics_supported(learned.move),
                    getattr(learned.move, "accuracy", 1.0),
                    getattr(learned.move, "priority", 0),
                    getattr(learned.move, "effect", None),
                    (
                        "special_attack"
                        if getattr(getattr(learned.move, "type", None), "kind", None) == "Special"
                        else "attack"
                    ),
                )
            )
        except (AttributeError, RuntimeError, TypeError, ValueError):
            continue

    switches = []
    for party_index, pokemon in enumerate(get_party()):
        if party_index == active.party_index or pokemon.is_egg or pokemon.current_hp <= 0:
            continue
        incoming = [value for name in opponent_moves if (value := damage(opponent, pokemon, name)) is not None]
        offensive = []
        for learned in pokemon.moves:
            if learned is not None and learned.pp > 0 and learned.move.base_power > 0:
                value = damage(pokemon, opponent, learned.move.name)
                if value is not None:
                    offensive.append(value)
        switches.append(
            PlannerSwitch(
                party_index,
                pokemon.name,
                pokemon.current_hp,
                pokemon.total_hp,
                max(incoming) if incoming else None,
                max(offensive, default=0),
            )
        )

    items = []
    try:
        for slot in get_item_bag().items:
            if slot.item.battle_use is ItemBattleUse.Healing:
                items.append(PlannerItem(slot.item, slot.quantity, slot.item.parameter))
        for slot in get_item_bag().poke_balls:
            items.append(PlannerItem(slot.item, slot.quantity, 0, True))
    except (AttributeError, RuntimeError, TypeError, ValueError):
        pass

    escape = 0.0
    if not battle_state.is_trainer_battle:
        try:
            escape = util.get_escape_chance()
        except (AttributeError, RuntimeError, TypeError, ValueError):
            escape = 0.0
    escape_confidence = (
        PlannerConfidence.HIGH
        if escape == 1
        else PlannerConfidence.MEDIUM if escape >= 0.5 else PlannerConfidence.INSUFFICIENT
    )
    planner_context = PlannerContext(
        active.species.name,
        active.current_hp,
        active.total_hp,
        opponent.species.name,
        opponent.current_hp,
        tuple(moves),
        tuple(switches),
        tuple(items),
        opponent_damage_max,
        tuple(opponent_moves),
        getattr(getattr(active, "stats", None), "speed", None),
        getattr(getattr(opponent, "stats", None), "speed", None),
        not battle_state.is_trainer_battle,
        battle_state.is_trainer_battle,
        util.can_switch(),
        escape > 0,
        escape_confidence,
        getattr(opponent, "total_hp", None),
        bool(getattr(battle_state, "nuzlocke_capture_target", False)),
        any(item.is_capture_ball and item.quantity > 0 for item in items),
        int(getattr(battle_state, "capture_appropriate_hp", 1) or 1),
        bool(getattr(battle_state, "capture_attempted", False)),
    )
    diagnostic_print(
        lambda: (
            "BATTLE_PLANNER_FACTS: "
            f"human_knowledge_policy={getattr(visible, 'policy', None)} "
            f"opponent_moves={[move.name for move in planner_context.opponent_moves]!r} "
            f"opponent_damage_max={planner_context.opponent_damage_max!r} "
            f"candidate_moves={[move.name for move in planner_context.moves]!r} "
            f"active_hp={planner_context.active_hp}/{planner_context.active_max_hp} "
            f"opponent_hp={planner_context.opponent_hp} "
            f"nuzlocke_capture_target={planner_context.capture_target} "
            f"capture_available={planner_context.capture_available}"
        ),
        trace=True,
    )
    decision = BattlePlanner().plan(planner_context)
    diagnostic_print(lambda: format_planner_decision_diagnostic(decision, planner_context), trace=True)
    return decision


def _mechanics_supported(move: Any) -> bool:
    """Effects not represented by the bounded evaluator must stay conservative."""
    unsupported_names = {"Bide", "Counter", "Mirror Coat", "Endeavor"}
    unsupported_effects = {"BIDE", "COUNTER", "MIRROR_COAT", "ENDEAVOR", "MULTI_HIT", "TWO_TURNS"}
    return (
        getattr(move, "name", None) not in unsupported_names
        and getattr(move, "effect", None) not in unsupported_effects
    )
