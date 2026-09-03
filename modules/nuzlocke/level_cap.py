"""Pure Emerald level-cap assessment and campaign battle-entry gateway."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, TYPE_CHECKING

from .campaign_state import CampaignFacts, FactStatus
from .emerald_campaign_registry import EmeraldBossDefinition, active_emerald_boss
from .rule_config import CampaignRuleId
from .rule_config import CampaignRulesConfig

if TYPE_CHECKING:
    from .campaign_state import CampaignState


@dataclass(frozen=True, slots=True)
class LevelCapAssessment:
    """Observed legality result for the active Emerald boss level cap."""

    rule_id: CampaignRuleId
    status: FactStatus
    active_boss: EmeraldBossDefinition | None
    level_cap: int | None
    illegal_party_indices: tuple[int, ...] = ()
    reason: str = ""

    @property
    def legal(self) -> bool:
        """Return whether the assessment is known and has no illegal members."""

        return self.status is FactStatus.KNOWN and not self.illegal_party_indices


def assess_level_cap(
    facts: CampaignFacts,
    party: Iterable[object] | None,
    *,
    objective_id: str | None = None,
) -> LevelCapAssessment:
    """Assess only the active leader objective; never infer a cap from levels."""
    progression = active_emerald_boss(facts)
    if progression.status is not FactStatus.KNOWN:
        return LevelCapAssessment(
            CampaignRuleId.LEVEL_CAP, progression.status, None, None, reason="boss completion is unavailable"
        )
    boss = progression.active_boss
    if boss is None:
        return LevelCapAssessment(CampaignRuleId.LEVEL_CAP, FactStatus.KNOWN, boss, boss.level_cap if boss else None)
    if party is None:
        return LevelCapAssessment(
            CampaignRuleId.LEVEL_CAP, FactStatus.UNKNOWN, boss, boss.level_cap, reason="party is unavailable"
        )
    illegal: list[int] = []
    for index, pokemon in enumerate(party):
        level = getattr(pokemon, "level", None)
        if not isinstance(level, int):
            return LevelCapAssessment(
                CampaignRuleId.LEVEL_CAP, FactStatus.UNKNOWN, boss, boss.level_cap, reason="party level is unavailable"
            )
        if level > boss.level_cap:
            illegal.append(getattr(pokemon, "party_index", index))
    return LevelCapAssessment(
        CampaignRuleId.LEVEL_CAP,
        FactStatus.KNOWN,
        boss,
        boss.level_cap,
        tuple(illegal),
        "party exceeds active leader cap" if illegal else "party is within active leader cap",
    )


def can_receive_experience_without_exceeding_cap(
    pokemon: object,
    experience_gain: int,
    level_cap: int,
) -> bool:
    """Return whether a known Pokémon can receive EXP and remain cap-legal.

    The live party exposes both total EXP and the species growth formula.  A
    current-level check alone is insufficient: a level-14 Pokémon can already
    be close enough to level 16 that one encounter would make it illegal for a
    level-15 leader.  Unknown EXP facts fail closed because this helper is used
    only to authorize an optional training participant.
    """

    if experience_gain < 0 or level_cap < 1:
        return False
    try:
        level = pokemon.level
        total_exp = pokemon.total_exp
        growth = pokemon.species.level_up_type
        if not isinstance(level, int) or not isinstance(total_exp, int):
            return False
        if level > level_cap:
            return False
        if level_cap >= 100:
            return True
        next_cap_exp = growth.get_experience_needed_for_level(level_cap + 1)
        return total_exp + experience_gain < next_cap_exp
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return False


class LevelCapRule:
    """Campaign-rule adapter for cap assessment."""

    rule_id = CampaignRuleId.LEVEL_CAP

    def evaluate(self, state: "CampaignState") -> LevelCapAssessment:
        """Assess the current party against the active boss cap."""

        return assess_level_cap(state.campaign_facts, state.party.value if state.party.is_known else None)

    def constrain(self, state: "CampaignState", candidate):
        """Leave non-battle campaign candidates unchanged."""

        return candidate


@dataclass(frozen=True, slots=True)
class BattleEntryDecision:
    """Conservative allow/deny result for entering a campaign battle."""

    allowed: bool
    reason: str
    assessment: LevelCapAssessment


def evaluate_battle_entry(
    facts: CampaignFacts,
    party: Iterable[object] | None,
    *,
    objective_id: str | None,
    rule_config: CampaignRulesConfig | None = None,
) -> BattleEntryDecision:
    """Single conservative gate used immediately before campaign battle action."""
    if rule_config is not None and not rule_config.is_enabled(CampaignRuleId.LEVEL_CAP):
        assessment = LevelCapAssessment(
            CampaignRuleId.LEVEL_CAP, FactStatus.KNOWN, None, None, reason="level-cap rule disabled"
        )
        return BattleEntryDecision(True, assessment.reason, assessment)
    assessment = assess_level_cap(facts, party, objective_id=objective_id)
    if assessment.status is not FactStatus.KNOWN:
        return BattleEntryDecision(False, f"level-cap legality is {assessment.status.value}", assessment)
    if assessment.illegal_party_indices:
        return BattleEntryDecision(False, assessment.reason, assessment)
    return BattleEntryDecision(True, assessment.reason or "no active level-cap constraint", assessment)
