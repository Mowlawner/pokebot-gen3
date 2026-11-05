"""Stable human-readable identities for battle diagnostics."""

from __future__ import annotations

from typing import Iterable, Any


def _party_member_for_battler(battler: Any, party: Iterable[Any] | None) -> Any | None:
    """Resolve the party record behind a live battle battler when available."""

    if battler is None or party is None:
        return None
    index = getattr(battler, "party_index", getattr(battler, "index", None))
    if not isinstance(index, int):
        return None
    try:
        return party[index]
    except (IndexError, KeyError, TypeError):
        try:
            return next(
                member
                for member in party
                if getattr(member, "index", None) == index
            )
        except (StopIteration, TypeError):
            return None


def battle_pokemon_identity(
    pokemon: Any,
    *,
    party: Iterable[Any] | None = None,
) -> tuple[str, str | None, int | None]:
    """Return ``(species, nickname, party_index)`` without speculative reads."""

    if pokemon is None:
        return "unknown", None, None
    party_member = _party_member_for_battler(pokemon, party)
    source = party_member or pokemon
    # The live battler is authoritative for species. The party record is used
    # for nickname lookup, but its slot can be stale during a replacement
    # prompt while the opponent party is being reordered.
    species_value = getattr(getattr(pokemon, "species", None), "name", None)
    if species_value is None:
        species_value = getattr(getattr(source, "species", None), "name", None)
    species = str(species_value or getattr(source, "name", None) or "unknown")
    nickname = getattr(source, "nickname", None)
    if nickname is None:
        nickname = getattr(pokemon, "nickname", None)
    nickname = str(nickname).strip() if nickname else None
    # PartyPokemon.name is its nickname or species fallback. Use it only when
    # it contains a name distinct from the species and no nickname property
    # was exposed by a lightweight test double.
    if not nickname:
        candidate_name = getattr(source, "name", None)
        if candidate_name and str(candidate_name).casefold() != species.casefold():
            nickname = str(candidate_name)
    index = getattr(pokemon, "party_index", getattr(source, "index", None))
    return species, nickname, index if isinstance(index, int) else None


def battle_pokemon_display_name(
    pokemon: Any,
    *,
    party: Iterable[Any] | None = None,
) -> str:
    """Render a battle identity with species and a custom nickname."""

    species, nickname, _ = battle_pokemon_identity(pokemon, party=party)
    if nickname and nickname.casefold() != species.casefold():
        return f'{species} "{nickname}"'
    return species


__all__ = ["battle_pokemon_display_name", "battle_pokemon_identity"]
