"""Stable identity for a valid Generation III Pokémon."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PokemonIdentity:
    """Persistent individual identity, independent of mutable game state."""

    personality_value: int
    original_trainer_id: int
    original_trainer_secret_id: int

    @classmethod
    def from_pokemon(cls, pokemon) -> "PokemonIdentity | None":
        """Build an identity only from a non-empty, checksum-valid Pokémon."""
        try:
            if pokemon.is_empty or not pokemon.is_valid:
                return None
            trainer = pokemon.original_trainer
            return cls(pokemon.personality_value, trainer.id, trainer.secret_id)
        except (AttributeError, IndexError, TypeError, ValueError):
            return None

    @classmethod
    def from_battle_pokemon(cls, pokemon) -> "PokemonIdentity | None":
        """Build identity from the persistent fields in BattlePokemon data."""
        try:
            personality_value = pokemon.personality_value
            ot_id = pokemon.ot_id
            if personality_value == 0 or ot_id == 0:
                return None
            return cls(personality_value, ot_id & 0xFFFF, (ot_id >> 16) & 0xFFFF)
        except (AttributeError, IndexError, TypeError, ValueError):
            return None
