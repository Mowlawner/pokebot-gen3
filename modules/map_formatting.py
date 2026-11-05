"""Shared human-readable formatting for ROM map identifiers."""

from __future__ import annotations

import re

from modules.map_data import MapFRLG, MapRSE


def map_display_name(map_id) -> str | None:
    """Return the enum's readable map name for a ROM map identifier."""

    if map_id is None:
        return None
    normalized = getattr(map_id, "value", map_id)
    pretty_name = getattr(map_id, "pretty_name", None)
    if pretty_name is not None:
        return pretty_name
    for map_enum in (MapRSE, MapFRLG):
        try:
            return map_enum(*normalized).pretty_name
        except (KeyError, TypeError, ValueError):
            continue
    return None


def format_map_id(map_id) -> str:
    """Format ``(group, number)`` with its ROM name beside the raw ID."""

    normalized = getattr(map_id, "value", map_id)
    name = map_display_name(map_id)
    if name is None:
        return repr(normalized)
    return f"{name} {normalized!r}"


def format_location(location) -> str:
    """Format a navigation location as ``MAP_NAME (id) @ (x, y)``."""

    if isinstance(location, tuple) and len(location) == 2:
        map_id, coordinates = location
        if isinstance(coordinates, tuple) and len(coordinates) == 2:
            return f"{format_map_id(map_id)} @ {coordinates!r}"
    return repr(location)


def format_map_ids_in_text(value) -> str:
    """Add names to map tuples without mistaking coordinates for maps.

    A bare ``(x, y)`` is ambiguous: it may be a map ID or a player
    coordinate, and many coordinate pairs happen to be valid ROM map IDs.
    Only format tuples in explicit map fields or the first component of a
    nested ``((map_group, map_number), (x, y))`` location.
    """

    text = value if isinstance(value, str) else repr(value)

    map_tuple = r"\((-?\d+),\s*(-?\d+)\)"

    def format_match(match):
        map_id = (int(match.group("group")), int(match.group("number")))
        name = map_display_name(map_id)
        return f"{name} {map_id!r}" if name is not None else match.group(0)

    # Explicit fields are unambiguous even when they are not part of a full
    # location, for example ``target_map=(0, 18)``.
    explicit = re.compile(
        r"(?P<prefix>\b(?:map|map_id|current_map|target_map|start_map|destination_map|"
        r"source_map|entry_map|from_map|to_map)\s*=\s*|"
        r"\btrainer(?:_id)?\s*:\s*)"
        rf"(?P<tuple>{map_tuple})"
    )

    def replace_explicit(match):
        tuple_match = re.fullmatch(map_tuple, match.group("tuple"))
        if tuple_match is None:
            return match.group(0)
        map_id = (int(tuple_match.group(1)), int(tuple_match.group(2)))
        name = map_display_name(map_id)
        return (
            f"{match.group('prefix')}{name} {map_id!r}"
            if name is not None
            else match.group(0)
        )

    text = explicit.sub(replace_explicit, text)

    # A nested location carries its own structure, so its first pair is the
    # map ID and the second pair is the coordinate. This also covers goal
    # reprs such as ``ReachLocation(((0, 10), (6, 17)))``.
    nested = re.compile(
        r"(?<=\()(?P<tuple>\((?P<group>-?\d+),\s*(?P<number>-?\d+)\))"
        r"(?=\s*,\s*\()"
    )
    text = nested.sub(format_match, text)

    return text


__all__ = ["format_location", "format_map_id", "format_map_ids_in_text", "map_display_name"]
