from collections import deque
from typing import Generator

from modules.context import context
from modules.debug import debug
from modules.memory import read_symbol


_BATTLE_MENU_CURSOR_TIMEOUT = 180


def _menu_cursor_path(
    current_index: int,
    target_index: int,
    valid_indices: tuple[int, ...] | None = None,
) -> tuple[str, ...]:
    """Return a safe directional path between battle-menu cursor positions.

    Emerald's move menu is a two-by-two grid, but the fourth position is
    absent when a Pokémon knows fewer than four moves.  A horizontal-first
    route can therefore try to enter the empty fourth position while moving
    from slot 2 to slot 1.  Search the actual occupied menu graph instead of
    assuming every grid position is usable.
    """

    if current_index == target_index:
        return ()

    occupied = set(range(4) if valid_indices is None else valid_indices)
    if target_index not in occupied:
        raise ValueError(f"Target menu index {target_index!r} is not an occupied position.")

    # Preserve the old horizontal-first preference where it is safe.  The
    # occupied-position filter makes the reverse three-move transition use
    # Up, Right rather than probing the empty slot 3.
    def neighbours(index: int):
        if index in (1, 3):
            yield index - 1, "Left"
        if index in (0, 2):
            yield index + 1, "Right"
        if index > 1:
            yield index - 2, "Up"
        if index < 2:
            yield index + 2, "Down"

    queue = deque([(current_index, ())])
    visited = {current_index}
    while queue:
        index, path = queue.popleft()
        for next_index, button in neighbours(index):
            if next_index not in occupied or next_index in visited:
                continue
            next_path = (*path, button)
            if next_index == target_index:
                return next_path
            visited.add(next_index)
            queue.append((next_index, next_path))

    raise RuntimeError(
        "Battle menu cursor has no legal path to its target: "
        f"current={current_index!r} target={target_index!r} occupied={tuple(sorted(occupied))!r}"
    )


def _select_in_menu(
    cursor_symbol: str,
    target_index: int,
    use_third_byte: bool = False,
    valid_indices: tuple[int, ...] | None = None,
) -> Generator:
    if target_index < 0 or target_index > 3:
        raise ValueError(f"Menu index must be a number between 0 and 3. '{target_index}' given.")

    first_iteration = True
    last_index = None
    for _ in range(_BATTLE_MENU_CURSOR_TIMEOUT):
        current_index = read_symbol(cursor_symbol, size=4)

        # In double battles, for the right-side Pokémon the 3rd byte of the index is used,
        # otherwise it's the 1st byte.
        if use_third_byte:
            current_index = current_index[2]
        else:
            current_index = current_index[0]

        if current_index == target_index:
            # If this function is called when the cursor is already pointing at the target,
            # in some edge cases we need to wait another frame until the UI becomes responsive.
            # This is _usually_ not necessary, but waiting that extra frame fixes the edge
            # case so here we go.
            if first_iteration:
                yield
            break
        path = _menu_cursor_path(current_index, target_index, valid_indices)
        context.emulator.press_button(path[0])
        yield
        yield
        first_iteration = False
        last_index = current_index
    else:
        frame = None
        try:
            frame = context.emulator.get_frame_count()
        except (AttributeError, RuntimeError, TypeError, ValueError):
            pass
        raise RuntimeError(
            "Battle menu cursor did not reach its target: "
            f"cursor={cursor_symbol!r} target={target_index!r} current={last_index!r} "
            f"frames={_BATTLE_MENU_CURSOR_TIMEOUT!r} emulator_frame={frame!r}"
        )


@debug.track
def scroll_to_battle_action(action_index: int) -> Generator:
    yield from _select_in_menu("gActionSelectionCursor", action_index)


@debug.track
def scroll_to_move(
    move_index: int,
    is_right_side_pokemon: bool = False,
    valid_indices: tuple[int, ...] | None = None,
) -> Generator:
    yield from _select_in_menu(
        "gMoveSelectionCursor",
        move_index,
        is_right_side_pokemon,
        valid_indices,
    )
