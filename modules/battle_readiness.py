"""Shared battle-readiness contract for campaign preparation decisions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from modules.battle_expectiminimax import SearchResult, search_planner_context


class BattleReadinessStatus(Enum):
    READY = "ready"
    PREPARE = "prepare"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class BattleReadiness:
    """A bounded assessment that recovery and navigation can consume."""

    checkpoint_id: str | None
    status: BattleReadinessStatus
    win_probability: float | None = None
    catastrophic_probability: float | None = None
    recommended_action: str | None = None
    required_hp_ratio: float | None = None
    expected_exp: int | None = None
    reason: str = ""
    search: SearchResult | None = None
    # Pre-battle planning facts.  ``lead_party_index`` is a current-party
    # slot; a negative value identifies a normalized PC candidate.
    lead_party_index: int | None = None
    lead_name: str | None = None
    lead_source: str = "party"
    lead_assessments: tuple = ()
    roster_action: str | None = None
    # Separate dimensions from the shared battle-risk contract. The legacy
    # ``catastrophic_probability`` field remains the party-wipe alias used by
    # older campaign diagnostics.
    active_faint_probability: float | None = None
    party_wipe_probability: float | None = None
    run_loss_probability: float | None = None
    # A roster plan maps each known opposing slot to a living party member.
    # It permits role-based coverage without requiring one Pokémon to solo
    # the entire trainer roster.
    roster_plan: tuple[tuple[int, int], ...] = ()
    roster_covered_opponents: int = 0
    roster_opponent_count: int = 0

    @property
    def ready(self) -> bool | None:
        if self.status is BattleReadinessStatus.READY:
            return True
        if self.status is BattleReadinessStatus.PREPARE:
            return False
        return None


def established_search_probability(search: SearchResult | None, attribute: str):
    """Return a probability only when the search established its outcome.

    A horizon or incomplete result can carry zero-valued internal counters
    because no terminal branch has been reached.  Those counters are useful
    to the evaluator, but are not evidence of a zero probability for the
    unfinished remainder of the battle.
    """

    if search is None:
        return None
    category = getattr(getattr(search, "category", None), "value", getattr(search, "category", None))
    if category != "terminal_complete" or not getattr(search, "complete", False):
        return None
    return getattr(search, attribute, None)


def assess_battle_readiness(
    checkpoint,
    search: SearchResult | None,
    *,
    required_hp_ratio: float = 0.5,
    expected_exp: int | None = None,
) -> BattleReadiness:
    """Convert a bounded search result into a stable campaign decision."""

    checkpoint_id = getattr(checkpoint, "checkpoint_id", None)
    if search is None:
        return BattleReadiness(
            checkpoint_id,
            BattleReadinessStatus.UNKNOWN,
            required_hp_ratio=required_hp_ratio,
            expected_exp=expected_exp,
            reason="no complete battle response envelope is available",
        )
    search_is_safe = bool(getattr(search, "is_safe", False))
    if search_is_safe or getattr(search, "is_autonomous_projection", False):
        reason = search.reason
        if not search_is_safe:
            reason = f"{reason}; executing the best available legal root while search remains unresolved"
        return BattleReadiness(
            checkpoint_id,
            BattleReadinessStatus.READY,
            established_search_probability(search, "win_probability"),
            established_search_probability(search, "catastrophic_probability"),
            search.action_name,
            required_hp_ratio,
            expected_exp,
            reason,
            search,
            active_faint_probability=established_search_probability(search, "active_faint_probability"),
            party_wipe_probability=established_search_probability(search, "party_wipe_probability"),
            run_loss_probability=established_search_probability(search, "run_loss_probability"),
        )
    return BattleReadiness(
        checkpoint_id,
        BattleReadinessStatus.PREPARE,
        established_search_probability(search, "win_probability"),
        established_search_probability(search, "catastrophic_probability"),
        search.action_name,
        required_hp_ratio,
        expected_exp,
        search.reason,
        search,
        active_faint_probability=established_search_probability(search, "active_faint_probability"),
        party_wipe_probability=established_search_probability(search, "party_wipe_probability"),
        run_loss_probability=established_search_probability(search, "run_loss_probability"),
    )


def assess_planner_context(checkpoint, planner_context, *, expected_exp: int | None = None) -> BattleReadiness:
    """Assess a currently observed checkpoint without reading the emulator."""

    return assess_battle_readiness(
        checkpoint,
        search_planner_context(planner_context),
        expected_exp=expected_exp,
    )
