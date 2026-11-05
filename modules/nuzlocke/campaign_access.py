"""ROM-owned access requirements used by generic campaign reachability.

The static map graph answers where the ROM is connected. This value object
describes the additional campaign capability a map requires. Definitions live
in the game-specific campaign registry; the reachability evaluator only
interprets observed facts.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CampaignAccessRequirement:
    """A capability that must be observed before a map can be entered."""

    requirement_id: str
    map_id: tuple[int, int]
    required_badges: tuple[str, ...] = ()
    required_facts: tuple[str, ...] = ()
    # Capability IDs are interpreted by CampaignState rather than by the
    # game-specific registry.  This keeps field-move and item access gates in
    # the same generic reachability contract as badge/story gates.
    required_capabilities: tuple[str, ...] = ()
    description: str = "campaign access requirement"
    # Source-derived requirements retain the evidence that authorized their
    # promotion.  These fields are optional so the hand-verified registry and
    # existing fixtures remain valid without migration.
    candidate_id: str | None = None
    provenance: tuple[tuple[str, str], ...] = ()
    confidence: str = "verified_registry"
    verification_status: str = "verified"


__all__ = ["CampaignAccessRequirement"]
