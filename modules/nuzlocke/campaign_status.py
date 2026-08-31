"""Human-readable projection of the semantic campaign state for the GUI."""

from __future__ import annotations

from dataclasses import dataclass

from modules.goals import Goal, SemanticTarget, SemanticTargetKind
from modules.map_data import MapRSE


@dataclass(frozen=True, slots=True)
class CampaignStatus:
    """Observational campaign status; it is never consulted by policy."""

    objective: str | None
    target: SemanticTarget | None
    intent: str
    tactical_goal: Goal | None = None
    complete: bool = False

    @classmethod
    def complete_status(cls) -> "CampaignStatus":
        """Create the terminal GUI status for a completed campaign slice."""

        return cls(None, None, "Campaign complete", complete=True)


def _target_text(target: SemanticTarget | None) -> str:
    """Render a semantic target compactly for human-readable status output."""

    if target is None:
        return "NONE"
    if target.kind is SemanticTargetKind.INTERACTION:
        return target.interaction_id or "interaction"
    if target.kind is SemanticTargetKind.LOCATION:
        return f"location {target.location}"
    map_name = next(
        (member.name.replace("_", " ").title() for member in MapRSE if member.value == target.target_map), None
    )
    if map_name is not None and map_name.startswith("Route") and map_name[5:].isdigit():
        map_name = f"Route {map_name[5:]}"
    return f"map {target.target_map}" + (f" ({map_name})" if map_name is not None else "")


def format_campaign_status(status: CampaignStatus | None) -> str:
    """Format the current semantic campaign status for the GUI."""

    if status is None:
        return "Objective: NONE\nTarget: NONE\nIntent: No semantic objective"
    if status.complete:
        return "Campaign: COMPLETE"
    objective = status.objective or "NONE"
    return f"Objective: {objective}\nTarget: {_target_text(status.target)}\nIntent: {status.intent}"


def recovery_status(target: SemanticTarget | None, intent: str) -> CampaignStatus:
    """Return the GUI projection for the currently executing recovery phase.

    Recovery is an execution interruption of the selected campaign objective,
    but it uses the same projection as ordinary campaign work so the GUI can
    show the concrete target and current intent.  The runtime is responsible
    for publishing this only after the corresponding observation boundary.
    """
    return CampaignStatus("Recover Party", target, intent)


__all__ = ["CampaignStatus", "format_campaign_status", "recovery_status"]
