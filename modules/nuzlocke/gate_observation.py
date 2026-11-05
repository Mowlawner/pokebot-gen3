"""Resolve source gate guards against an already captured campaign snapshot."""

from __future__ import annotations

import re
from dataclasses import replace

from .gate_discovery import GateCandidate
from .snapshots import CampaignObservationLifecycle, CampaignObservationSnapshot


_COMMAND_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s+([^,\s]+)(?:\s*,\s*([^,\s]+))?")


def _observations_available(snapshot: CampaignObservationSnapshot) -> str | None:
    if not snapshot.available or snapshot.lifecycle is CampaignObservationLifecycle.UNAVAILABLE:
        return "unavailable"
    if snapshot.lifecycle is CampaignObservationLifecycle.FRESH_START:
        return "unavailable"
    return None


def _flag_values(snapshot: CampaignObservationSnapshot) -> dict[str, bool]:
    return {flag.name: flag.value for flag in snapshot.flags}


def _variable_values(snapshot: CampaignObservationSnapshot) -> dict[str, int]:
    return {variable.name: variable.value for variable in snapshot.variables}


def resolve_guard_state(candidate: GateCandidate, snapshot: CampaignObservationSnapshot) -> str:
    """Return ``satisfied``, ``unsatisfied``, ``unknown``, or ``unavailable``."""

    unavailable = _observations_available(snapshot)
    if unavailable is not None:
        return unavailable
    symbol = candidate.guard_symbol
    if not symbol:
        return "unknown"

    flags = _flag_values(snapshot)
    variables = _variable_values(snapshot)
    if symbol in flags:
        value: object = flags[symbol]
    elif symbol.removeprefix("FLAG_") in flags:
        # Source scripts use the C macro spelling (FLAG_FOO), while the
        # runtime event-flag table is keyed by the save-reader spelling
        # (FOO). Keep the candidate's exact source symbol in provenance while
        # resolving both representations at the observation boundary.
        value = flags[symbol.removeprefix("FLAG_")]
    elif symbol in variables:
        value = variables[symbol]
    else:
        return "unknown"

    expression = candidate.guard_expression or ""
    if " is set" in expression:
        satisfied = bool(value)
    elif " is unset" in expression:
        satisfied = not bool(value)
    else:
        command_match = _COMMAND_RE.match(expression)
        if command_match is None:
            return "unknown"
        command, _, raw_comparison = command_match.groups()
        if command in {"checkflag", "goto_if_set", "call_if_set"}:
            satisfied = bool(value)
        elif command in {"goto_if_unset", "call_if_unset"}:
            satisfied = not bool(value)
        elif command in {"goto_if_defeated", "goto_if_not_defeated"}:
            # Trainer clearance is a separate observation source and cannot be
            # inferred from an event flag or variable with the same name.
            return "unknown"
        elif command in {
            "goto_if_eq",
            "goto_if_ne",
            "goto_if_lt",
            "goto_if_le",
            "goto_if_gt",
            "goto_if_ge",
            "call_if_eq",
            "call_if_ne",
            "call_if_lt",
            "call_if_le",
            "call_if_gt",
            "call_if_ge",
            "map_script_2",
        }:
            if raw_comparison is None:
                return "unknown"
            try:
                comparison = int(raw_comparison, 0)
                numeric_value = int(value)
            except (TypeError, ValueError):
                return "unknown"
            operator = command.rsplit("_", 1)[-1]
            if command == "map_script_2":
                operator = "eq"
            satisfied = {
                "eq": numeric_value == comparison,
                "ne": numeric_value != comparison,
                "lt": numeric_value < comparison,
                "le": numeric_value <= comparison,
                "gt": numeric_value > comparison,
                "ge": numeric_value >= comparison,
            }.get(operator, False)
        else:
            return "unknown"
    return "satisfied" if satisfied else "unsatisfied"


def resolve_candidate_observation(
    candidate: GateCandidate,
    snapshot: CampaignObservationSnapshot,
) -> GateCandidate:
    """Return an immutable candidate with the current guard state attached."""

    return replace(candidate, current_state=resolve_guard_state(candidate, snapshot))


__all__ = ["resolve_candidate_observation", "resolve_guard_state"]
