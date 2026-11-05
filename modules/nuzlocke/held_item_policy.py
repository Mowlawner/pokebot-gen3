"""Policy for deciding which held items are worth keeping on the party."""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class HeldItemDecision:
    retain: bool
    reason: str


def evaluate_held_item(pokemon: Any, *, planned_item_names: tuple[str, ...] = ()) -> HeldItemDecision:
    """Classify one held item without opening a menu or mutating the party.

    Items with a real Gen III hold effect are useful by default: this includes
    healing/status berries and training or battle items.  Incidental held items
    with no effect, including items found on caught wild Pokémon, are returned
    to the bag.  A caller may name an otherwise inert item explicitly when a
    later campaign action has a reason to preserve it.
    """

    item = getattr(pokemon, "held_item", None)
    if item is None:
        return HeldItemDecision(True, "no held item")

    item_name = getattr(item, "name", str(item))
    if item_name in planned_item_names:
        return HeldItemDecision(True, f"{item_name} is reserved for a planned campaign use")

    ability = getattr(getattr(pokemon, "ability", None), "name", None)
    if ability == "Pickup":
        return HeldItemDecision(False, f"{item_name} is a Pickup result and should be collected")

    hold_effect = getattr(item, "hold_effect", None)
    hold_effect_name = getattr(hold_effect, "name", None) or getattr(hold_effect, "value", None)
    if hold_effect_name and str(hold_effect_name).casefold() not in {"noeffect", "no_effect"}:
        return HeldItemDecision(True, f"{item_name} has strategic hold effect {hold_effect_name}")
    return HeldItemDecision(False, f"{item_name} has no strategic hold effect")


def held_item_needs_removal(pokemon: Any, *, planned_item_names: tuple[str, ...] = ()) -> bool:
    """Return whether a party Pokémon should have its current item removed."""

    return not evaluate_held_item(pokemon, planned_item_names=planned_item_names).retain


def party_held_items_needing_removal(party: Any) -> tuple[int, ...]:
    """Return stable party indices whose incidental item should be collected."""

    return tuple(
        index
        for index, pokemon in enumerate(party)
        if not getattr(pokemon, "is_egg", False) and held_item_needs_removal(pokemon)
    )
