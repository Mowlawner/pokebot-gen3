"""Campaign-level reachability for optional objectives.

The ROM map graph describes physical connections, but a physical connection is
not necessarily an executable campaign route.  Story gates, field-move
requirements, and mandatory trainer checkpoints can sit between the current
map and a future encounter.  This module overlays the currently authoritative
campaign facts on the cached map graph and reports whether a destination can
be mounted now.

This is deliberately a conservative planning boundary.  A registered
mandatory checkpoint with an unresolved clearance fact remains a preparation
blocker.  The encounter may remain visible as a future opportunity, but it is
not allowed to take tactical ownership until the blocker has been cleared.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from modules.world_navigation import WorldMapGraph, WorldNavigationError, WorldRoute, get_world_map_graph

from .campaign_state import CampaignState, FactStatus
from .campaign_access import CampaignAccessRequirement
from modules.trainer_checkpoints import KNOWN_TRAINER_CHECKPOINTS, TrainerCheckpoint


class CampaignReachabilityStatus(Enum):
    """Whether a campaign destination can be pursued at the current boundary."""

    REACHABLE_NOW = "reachable_now"
    REACHABLE_AFTER_PREPARATION = "reachable_after_preparation"
    UNREACHABLE = "unreachable"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class CampaignReachability:
    """A route assessment with the blockers needed to make it executable."""

    destination: tuple[int, int]
    status: CampaignReachabilityStatus
    route: WorldRoute | None = None
    blockers: tuple[str, ...] = ()
    checkpoint_ids: tuple[str, ...] = ()
    access_requirement_ids: tuple[str, ...] = ()

    @property
    def executable_now(self) -> bool:
        """Return whether this destination may own an immediate objective."""

        return self.status is CampaignReachabilityStatus.REACHABLE_NOW


def _clearance_status(state: CampaignState, checkpoint: TrainerCheckpoint) -> FactStatus | None:
    """Return the status of a registered checkpoint's clearance fact."""

    fact_name = getattr(checkpoint, "clearance_fact", None)
    if not fact_name:
        return None
    fact = getattr(state.campaign_facts, fact_name, None)
    return getattr(fact, "status", None)


def _checkpoint_is_cleared(state: CampaignState, checkpoint: TrainerCheckpoint) -> bool:
    """Return whether campaign facts prove that a checkpoint is behind us."""

    fact_name = getattr(checkpoint, "clearance_fact", None)
    if not fact_name:
        return False
    fact = getattr(state.campaign_facts, fact_name, None)
    return bool(fact is not None and fact.status is FactStatus.KNOWN and fact.value is True)


def _checkpoint_blocks_route(
    checkpoint: TrainerCheckpoint,
    route: WorldRoute,
    current_map: tuple[int, int],
    destination: tuple[int, int],
) -> bool:
    """Return whether a checkpoint lies before the requested destination.

    A checkpoint's own map remains a valid encounter destination: the player
    can acquire an encounter before engaging the trainer.  The checkpoint
    blocks a route when it is an intermediate map, or when the player is
    already on that map and is trying to leave it for another destination.
    """

    if checkpoint.map_id not in route.maps:
        return False
    if checkpoint.map_id == destination:
        return False
    # A checkpoint on the current map is evidence about one local hazard, not
    # proof that every other destination is behind that hazard.  Navigation
    # and route analysis decide whether the requested path crosses its sight
    # line.  Campaign reachability only applies checkpoints that are on an
    # intermediate map in the ROM-backed route.
    if checkpoint.map_id == current_map:
        return False
    return checkpoint.map_id in route.maps[:-1]


def _access_requirement_blocks_route(
    requirement: CampaignAccessRequirement,
    route: WorldRoute,
    current_map: tuple[int, int],
) -> bool:
    """Return whether a map capability is needed before this route."""

    # Being on a map proves that its entry requirement has already been
    # crossed. A future entry into that map, including arrival at the target,
    # must satisfy the registered requirement first.
    return requirement.map_id in route.maps and requirement.map_id != current_map


def _access_requirement_blockers(
    state: CampaignState,
    requirement: CampaignAccessRequirement,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return (known-missing, unresolved) blockers for one requirement."""

    blockers: list[str] = []
    unresolved: list[str] = []
    for badge_name in requirement.required_badges:
        badge = state.has_badge(badge_name)
        if badge.status is FactStatus.KNOWN and badge.value is True:
            continue
        if badge.status is FactStatus.KNOWN:
            blockers.append(f"{requirement.requirement_id}: missing badge {badge_name}")
        else:
            unresolved.append(f"{requirement.requirement_id}: {badge_name} is unresolved")
    for fact_name in requirement.required_facts:
        fact = getattr(state.campaign_facts, fact_name, None)
        status = getattr(fact, "status", None)
        value = getattr(fact, "value", None)
        if status is FactStatus.KNOWN and value is True:
            continue
        if status is FactStatus.KNOWN:
            blockers.append(f"{requirement.requirement_id}: missing campaign fact {fact_name}")
        else:
            unresolved.append(f"{requirement.requirement_id}: {fact_name} is unresolved")
    for capability_name in requirement.required_capabilities:
        capability = state.campaign_facts.capability(capability_name)
        if capability.status is FactStatus.KNOWN and capability.value is True:
            continue
        if capability.status is FactStatus.KNOWN:
            blockers.append(f"{requirement.requirement_id}: missing capability {capability_name}")
        else:
            unresolved.append(f"{requirement.requirement_id}: {capability_name} is unresolved")
    return tuple(blockers), tuple(unresolved)


def _default_access_requirements() -> tuple[CampaignAccessRequirement, ...]:
    """Load the active game's registry without creating an import cycle."""

    try:
        from modules.context import context

        rom = getattr(context, "rom", None)
        if rom is not None and not getattr(rom, "is_rse", False):
            return ()
        from .emerald_campaign_registry import emerald_map_access_requirements

        return emerald_map_access_requirements()
    except (AttributeError, ImportError, RuntimeError, TypeError, ValueError):
        return ()


def assess_campaign_reachability(
    state: CampaignState,
    destination: tuple[int, int],
    *,
    graph: WorldMapGraph | None = None,
    checkpoints: tuple[TrainerCheckpoint, ...] = KNOWN_TRAINER_CHECKPOINTS,
    access_requirements: tuple[CampaignAccessRequirement, ...] | None = None,
) -> CampaignReachability:
    """Assess whether ``destination`` is executable from the current state.

    The topology remains ROM-derived and cached by :func:`get_world_map_graph`.
    Dynamic checkpoint clearance is evaluated from the immutable campaign
    state, so callers can cache the result using their normal campaign
    planning signature.  Unknown checkpoint facts are handled conservatively
    as preparation blockers rather than being treated as an open route.
    """

    if not state.raw_map.is_known:
        return CampaignReachability(destination, CampaignReachabilityStatus.UNKNOWN, blockers=("current map unknown",))

    current_map = state.raw_map.value
    if current_map == destination:
        return CampaignReachability(destination, CampaignReachabilityStatus.REACHABLE_NOW)

    try:
        route_graph = graph or get_world_map_graph()
        route = route_graph.route(current_map, destination)
    except (AttributeError, RuntimeError, TypeError, ValueError, WorldNavigationError):
        return CampaignReachability(
            destination,
            CampaignReachabilityStatus.UNREACHABLE,
            blockers=(f"no ROM map route from {current_map!r} to {destination!r}",),
        )

    blockers: list[str] = []
    checkpoint_ids: list[str] = []
    for checkpoint in checkpoints:
        if not getattr(checkpoint, "mandatory", False):
            continue
        if not _checkpoint_blocks_route(checkpoint, route, current_map, destination):
            continue
        if _checkpoint_is_cleared(state, checkpoint):
            continue
        checkpoint_ids.append(checkpoint.checkpoint_id)
        fact_name = getattr(checkpoint, "clearance_fact", None)
        status = _clearance_status(state, checkpoint)
        if status is FactStatus.UNKNOWN or status is FactStatus.UNAVAILABLE or fact_name is None:
            blockers.append(f"{checkpoint.checkpoint_id}: checkpoint clearance is unresolved")
        else:
            blockers.append(f"{checkpoint.checkpoint_id}: mandatory trainer preparation required")

    access_requirement_ids: list[str] = []
    unresolved_access: list[str] = []
    requirements = _default_access_requirements() if access_requirements is None else access_requirements
    for requirement in requirements:
        if not _access_requirement_blocks_route(requirement, route, current_map):
            continue
        requirement_blockers, requirement_unresolved = _access_requirement_blockers(state, requirement)
        if requirement_blockers or requirement_unresolved:
            access_requirement_ids.append(requirement.requirement_id)
            blockers.extend(requirement_blockers)
            unresolved_access.extend(requirement_unresolved)

    if blockers:
        return CampaignReachability(
            destination,
            CampaignReachabilityStatus.REACHABLE_AFTER_PREPARATION,
            route,
            tuple(blockers),
            tuple(checkpoint_ids),
            tuple(access_requirement_ids),
        )
    if unresolved_access:
        return CampaignReachability(
            destination,
            CampaignReachabilityStatus.UNKNOWN,
            route,
            tuple(unresolved_access),
            tuple(checkpoint_ids),
            tuple(access_requirement_ids),
        )
    return CampaignReachability(destination, CampaignReachabilityStatus.REACHABLE_NOW, route)


__all__ = [
    "CampaignReachability",
    "CampaignReachabilityStatus",
    "CampaignAccessRequirement",
    "assess_campaign_reachability",
]
