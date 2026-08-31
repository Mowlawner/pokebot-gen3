"""Reusable, gender-aware nickname policy for observed Pokémon."""

from __future__ import annotations

from typing import Literal

POKEMON_NICKNAME_MAX_LENGTH = 10
PokemonGender = Literal["male", "female", None]


def _human_name(gender: PokemonGender) -> str:
    """Request one human name from the configured name provider."""

    import names

    return names.get_first_name() if gender is None else names.get_first_name(gender=gender)


def generate_pokemon_nickname(
    species: str,
    gender: PokemonGender,
    *,
    max_length: int = POKEMON_NICKNAME_MAX_LENGTH,
) -> str:
    """Generate a valid human nickname; species is retained for future policy."""
    del species
    if max_length < 1:
        raise ValueError("max_length must be positive")
    # Names normally fit quickly. Keep a finite safety fallback for a broken
    # or unusually constrained provider without handing an invalid name to ROM.
    for _ in range(100):
        name = _human_name(gender)
        if 0 < len(name) <= max_length:
            return name
    name = _human_name(gender)
    if not name:
        raise ValueError("The Names library returned an empty Pokémon nickname")
    return name[:max_length]


__all__ = ["POKEMON_NICKNAME_MAX_LENGTH", "PokemonGender", "generate_pokemon_nickname"]
