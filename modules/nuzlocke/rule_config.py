"""Run configuration and extension contracts for campaign rules.

This module is intentionally independent of emulator access, event reduction,
and profile I/O.  Configuration loaders translate their representation into
``CampaignRulesConfig`` at the application boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Protocol, TypeVar, runtime_checkable

if TYPE_CHECKING:
    from .campaign_state import CampaignState


class CampaignRuleId(str, Enum):
    """Stable identifiers suitable for profile/run configuration."""

    ONE_ENCOUNTER_PER_AREA = "one_encounter_per_area"
    FAINTING = "fainting"
    SPECIES_CLAUSE = "species_clause"
    LEVEL_CAP = "level_cap"


DEFAULT_NUZLOCKE_RULES = frozenset(
    {
        CampaignRuleId.ONE_ENCOUNTER_PER_AREA,
        CampaignRuleId.FAINTING,
        CampaignRuleId.LEVEL_CAP,
    }
)


@dataclass(frozen=True, slots=True)
class CampaignRulesConfig:
    """The enabled campaign rules for one run.

    The default is deliberately compatible with the existing Nuzlocke runtime.
    Use :meth:`unrestricted` for a campaign run with no Nuzlocke constraints.
    """

    enabled_rules: frozenset[CampaignRuleId] = DEFAULT_NUZLOCKE_RULES

    @classmethod
    def from_names(cls, names: tuple[str, ...] | list[str]) -> "CampaignRulesConfig":
        """Build configuration from serialized rule identifier names."""

        return cls(frozenset(CampaignRuleId(name) for name in names))

    @classmethod
    def unrestricted(cls) -> "CampaignRulesConfig":
        """Build configuration with every optional campaign rule disabled."""

        return cls(frozenset())

    def is_enabled(self, rule: CampaignRuleId) -> bool:
        """Return whether the supplied rule is enabled for this run."""

        return rule in self.enabled_rules


RuleConstraint = TypeVar("RuleConstraint")


@runtime_checkable
class CampaignRule(Protocol[RuleConstraint]):
    """Pure extension point for rules that constrain campaign decisions.

    Concrete rules may return any immutable assessment type.  They must not
    read emulator state or mutate objectives; planners apply their returned
    constraints when selecting an objective or battle action.
    """

    rule_id: CampaignRuleId

    def evaluate(self, state: "CampaignState") -> RuleConstraint:
        """Assess the already-materialized campaign state."""

    def constrain(self, state: "CampaignState", candidate: RuleConstraint) -> RuleConstraint:
        """Return a constrained candidate without mutating it."""
