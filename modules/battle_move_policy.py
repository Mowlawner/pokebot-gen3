"""Explainable move-learning policy used by autonomous battle strategies."""

from __future__ import annotations

from dataclasses import dataclass


_CAPTURE_STATUS_NAMES = frozenset({"SLEEP", "PARALYZE", "PARALYSIS", "YAWN"})
_UNSAFE_DAMAGE_OVER_TIME = frozenset({"POISON", "TOXIC", "BURN", "LEECH_SEED", "LEECHSEEDED"})


@dataclass(frozen=True, slots=True)
class MoveSetDecision:
    """Result of comparing the old and new four-move sets."""

    replacement_index: int
    reason: str
    current_score: tuple[float, ...]
    candidate_score: tuple[float, ...]


def _move_name(move) -> str:
    return str(getattr(move, "name", ""))


def _effect_name(move) -> str:
    return str(getattr(move, "effect", "") or "").upper()


def _is_hm(move) -> bool:
    item = getattr(move, "tm_hm", None)
    return str(getattr(item, "name", "")).upper().startswith("HM")


def _is_damage_over_time(move) -> bool:
    effect = _effect_name(move)
    name = _move_name(move).upper().replace(" ", "_")
    return effect in _UNSAFE_DAMAGE_OVER_TIME or name in _UNSAFE_DAMAGE_OVER_TIME


def _has_capture_status(move) -> bool:
    effect = _effect_name(move)
    name = _move_name(move).upper().replace(" ", "_")
    return effect in _CAPTURE_STATUS_NAMES or name in _CAPTURE_STATUS_NAMES


def _is_damaging(move) -> bool:
    return int(getattr(move, "base_power", 0) or 0) > 0


def _accuracy(move) -> float:
    try:
        return max(0.0, min(1.0, float(getattr(move, "accuracy", 1.0))))
    except (TypeError, ValueError):
        return 0.0


def _move_value(move, species_types=()) -> float:
    """Estimate general value while keeping reliability ahead of raw power."""

    if move is None:
        return -1000.0
    if _is_hm(move):
        return 10000.0
    accuracy = _accuracy(move)
    power = float(getattr(move, "base_power", 0) or 0)
    move_type = getattr(getattr(move, "type", None), "name", None)
    stab = 1.5 if any(getattr(item, "name", None) == move_type for item in species_types) else 1.0
    if power > 0:
        # A damaging move with a secondary effect remains primarily a damage
        # move. The secondary effect is a small bonus rather than its main
        # source of value.
        value = power * accuracy * stab
        if _is_damage_over_time(move):
            # Damage-over-time is a risk penalty, not a reason to pretend the
            # move has no direct damage. Capture policy separately rejects
            # poison, toxic, and burn when it could endanger the target.
            value -= 20.0
        value += min(8.0, float(getattr(move, "secondary_accuracy", 0.0) or 0.0) * 8.0)
    elif _has_capture_status(move):
        value = 24.0 * _accuracy(move)
    elif _effect_name(move):
        value = 8.0 * _accuracy(move)
    else:
        value = 0.0
    value += min(5.0, float(getattr(move, "pp", 0) or 0) / 5.0)
    return value


def _set_score(moves, species_types=()) -> tuple[float, ...]:
    present = tuple(move for move in moves if move is not None)
    damaging = tuple(move for move in present if _is_damaging(move))
    types = {getattr(getattr(move, "type", None), "name", None) for move in damaging}
    reliable_damage = sum(_move_value(move, species_types) for move in damaging)
    capture_status = sum(1 for move in present if _has_capture_status(move))
    utility = sum(1 for move in present if not _is_damaging(move) and not _is_damage_over_time(move))
    unsafe_dot = sum(1 for move in present if _is_damage_over_time(move))
    strongest = max((_move_value(move, species_types) for move in damaging), default=0.0)
    return (
        float(len(damaging)),
        float(len(types)),
        float(capture_status),
        float(utility),
        reliable_damage,
        strongest,
        -float(unsafe_dot),
    )


def choose_move_replacement(
    pokemon,
    new_move,
    *,
    banned_moves=(),
    protected_indices=(),
) -> MoveSetDecision:
    """Choose the replacement that improves the complete move-set score.

    Returning index 4 means that the move should be declined. Empty slots are
    accepted immediately when the move is legal. HM slots are protected unless
    the caller explicitly removes them from ``protected_indices``.
    """

    banned = frozenset(banned_moves)
    existing = tuple(getattr(pokemon, "moves", ()) or ())
    existing_moves = tuple(getattr(item, "move", None) if item is not None else None for item in existing)
    if new_move is None or _move_name(new_move) in banned:
        return MoveSetDecision(4, f"declined {_move_name(new_move) or 'unknown move'} because it is banned", (), ())
    if any(_move_name(move) == _move_name(new_move) for move in existing_moves if move is not None):
        return MoveSetDecision(4, f"declined duplicate move {_move_name(new_move)}", (), ())

    species_types = tuple(getattr(getattr(pokemon, "species", None), "types", ()) or ())
    current = _set_score(existing_moves, species_types)
    candidates: list[tuple[tuple[float, ...], int, str]] = []
    for index in range(4):
        old_move = existing_moves[index] if index < len(existing_moves) else None
        if index in protected_indices or (old_move is not None and _is_hm(old_move)):
            continue
        candidate_moves = list(existing_moves[:4])
        while len(candidate_moves) < 4:
            candidate_moves.append(None)
        candidate_moves[index] = new_move
        score = _set_score(tuple(candidate_moves), species_types)
        if old_move is None:
            candidates.append((score, index, f"filled empty move slot with {_move_name(new_move)}"))
        elif score > current:
            candidates.append((score, index, f"replaced {_move_name(old_move)} with {_move_name(new_move)}"))

    if not candidates:
        return MoveSetDecision(
            4,
            f"declined {_move_name(new_move)} because no legal replacement improves the complete move set",
            current,
            current,
        )
    candidate_score, index, explanation = max(candidates, key=lambda item: (item[0], -item[1]))
    return MoveSetDecision(index, explanation, current, candidate_score)


def move_policy_summary(move) -> str:
    """Expose the safety role used by diagnostics and tests."""

    if _is_damage_over_time(move):
        return "damage_over_time_unsafe_for_capture"
    if _is_damaging(move):
        return "damaging" if not _has_capture_status(move) else "damaging_with_secondary_status"
    if _has_capture_status(move):
        return "capture_status"
    return "utility"
