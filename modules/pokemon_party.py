from typing import Generator

from modules.context import context
from modules.memory import read_symbol, get_event_var
from modules.pokemon import Pokemon, Move, Species, Ability
from modules.state_cache import state_cache
from modules.tasks import get_global_script_context, get_task

_POKEMON_CENTER_RETURN_SCRIPT = "EventScript_PkmnCenterNurse_ReturnPkmn"


def _party_read_is_during_nurse_message_render() -> bool:
    """Avoid speculative emulation while Emerald renders the nurse return text.

    A party slot can be transiently invalid while Emerald restores the party.
    The normal retry path peeks one native frame ahead, but that frame can
    re-enter the known mGBA render stall.  The controller will own the next
    fresh-B pulse; this read must remain passive until then.
    """
    try:
        if not context.rom.is_emerald:
            return False
        script_context = get_global_script_context()
        task = get_task("Task_DrawFieldMessage")
        return bool(
            script_context is not None
            and getattr(script_context, "is_active", False)
            and getattr(script_context, "script_function_name", None) == _POKEMON_CENTER_RETURN_SCRIPT
            and getattr(script_context, "native_function_name", None) == "IsFieldMessageBoxHidden"
            and task is not None
            and task.data_value(0) == 2
        )
    except (AttributeError, RuntimeError, TypeError, ValueError, IndexError):
        return False


class PartyPokemon(Pokemon):
    def __init__(self, data: bytes, index: int):
        super().__init__(data)
        self.index = index


class Party:
    def __init__(self, pokemon: list[PartyPokemon]):
        self._pokemon = pokemon

    def __eq__(self, other):
        if isinstance(other, Party):
            return len(other._pokemon) == len(self._pokemon) and all(
                [other[n] == self[n] for n in range(len(self._pokemon))]
            )
        else:
            return NotImplemented

    def __ne__(self, other):
        if isinstance(other, Party):
            return len(other._pokemon) != len(self._pokemon) or any(
                [other[n] != self[n] for n in range(len(self._pokemon))]
            )
        else:
            return NotImplemented

    def __contains__(self, item):
        if isinstance(item, Pokemon):
            return item in self._pokemon
        elif isinstance(item, Species):
            return any([pokemon.species is item for pokemon in self._pokemon])
        else:
            return NotImplemented

    def __iter__(self) -> Generator[PartyPokemon, None, None]:
        yield from self._pokemon

    def __len__(self) -> int:
        return len(self._pokemon)

    def __getitem__(self, item: int | slice):
        return self._pokemon[item]

    @property
    def contains_eggs(self) -> bool:
        return any([pokemon.is_egg for pokemon in self._pokemon])

    @property
    def eggs(self) -> list[PartyPokemon]:
        return [pokemon for pokemon in self._pokemon if pokemon.is_egg]

    @property
    def non_eggs(self) -> list[PartyPokemon]:
        return [pokemon for pokemon in self._pokemon if not pokemon.is_egg]

    @property
    def non_fainted_pokemon(self) -> list[PartyPokemon]:
        return list(
            filter(
                lambda pokemon: not pokemon.is_egg and pokemon.current_hp > 0,
                self._pokemon,
            )
        )

    @property
    def first_non_fainted(self) -> PartyPokemon | None:
        non_fainted_pokemon = self.non_fainted_pokemon
        if len(non_fainted_pokemon) > 0:
            return self.non_fainted_pokemon[0]
        else:
            return None

    def has_pokemon_with_move(self, move: Move | str, with_pp_remaining: bool = False) -> bool:
        return any(pokemon.knows_move(move, with_pp_remaining) and not pokemon.is_egg for pokemon in self._pokemon)

    def first_pokemon_with_move(self, move: Move | str, with_pp_remaining: bool = False) -> PartyPokemon | None:
        for pokemon in self._pokemon:
            if not pokemon.is_egg and pokemon.knows_move(move, with_pp_remaining):
                return pokemon
        return None

    def has_pokemon_with_ability(self, ability: Ability | str) -> bool:
        ability_name = ability.name if isinstance(ability, Ability) else str(ability)
        return any(pokemon.ability.name == ability_name and not pokemon.is_egg for pokemon in self._pokemon)

    def first_pokemon_with_ability(self, ability: Ability | str) -> PartyPokemon | None:
        ability_name = ability.name if isinstance(ability, Ability) else str(ability)
        for pokemon in self._pokemon:
            if not pokemon.is_egg and pokemon.ability.name == ability_name:
                return pokemon
        return None

    def get_index_for_pokemon(self, pokemon: Pokemon) -> int:
        for party_index, party_pokemon in enumerate(self._pokemon):
            if pokemon.data[:4] == party_pokemon.data[:4]:
                return party_index
        raise RuntimeError("This Pokémon is not in the player's party.")

    def to_list(self) -> list[dict]:
        return [pokemon.to_dict() for pokemon in self._pokemon]


def get_party_size() -> int:
    return len(get_party())


def get_party() -> Party:
    """
    :return: The player's party of Pokémon.
    """

    if state_cache.party.age_in_frames == 0:
        return state_cache.party.value

    def read_party_pokemon(party_index: int) -> PartyPokemon | None:
        party_pokemon = PartyPokemon(read_symbol("gPlayerParty", offset=party_index * 100, size=100), party_index)
        return party_pokemon if not party_pokemon.is_empty and party_pokemon.is_valid else None

    list_of_pokemon = []
    number_of_pokemon_in_party = read_symbol("gPlayerPartyCount", size=1)[0]
    for index in range(number_of_pokemon_in_party):
        pokemon = read_party_pokemon(index)

        # It's possible for party data to be written while we are trying to read it, in which case
        # the checksum would be wrong and `parse_pokemon()` returns `None`.
        #
        # In order to still get a valid result, we will 'peek' into next frame's memory by
        # (1) advancing the emulation by one frame, (2) reading the memory, (3) restoring the previous
        # frame's state, so we don't mess with frame accuracy.
        if pokemon is None and state_cache.party.value is not None and _party_read_is_during_nurse_message_render():
            # Returning the last valid snapshot is safe for the frame-level
            # observers and, importantly, keeps this path from calling
            # peek_frame() while the nurse's message printer is rendering.
            return state_cache.party.value

        if pokemon is None:
            retries = 5
            with context.emulator.peek_frame():
                while retries > 0 and pokemon is None:
                    retries -= 1
                    pokemon = read_party_pokemon(index)
                    if pokemon is None:
                        context.emulator._core.run_frame()
        if pokemon is None:
            if read_symbol("gPlayerParty", offset=index * 100, size=100).count(b"\x00") >= 99:
                continue
            else:
                raise RuntimeError(f"Party Pokémon #{index + 1} was invalid for two frames in a row.")

        list_of_pokemon.append(pokemon)

    party = Party(list_of_pokemon)
    state_cache.party = party

    return party


def get_opponent_party() -> Party | None:
    """
    Gets the opponent's party (obviously only makes sense to check when in a battle.)
    :return: The full party of the opponent, or `None` if there is no valid opponent at the moment.
    """
    if state_cache.opponent.age_in_frames == 0:
        return state_cache.opponent.value

    list_of_pokemon = []
    data = read_symbol("gEnemyParty")
    for index in range(6):
        offset = index * 100
        pokemon = PartyPokemon(data[offset : offset + 100], index)
        if pokemon is None:
            if index == 0:
                return None
            else:
                continue
        list_of_pokemon.append(pokemon)

    party = Party(list_of_pokemon)
    state_cache.opponent = party

    return party


def get_current_repel_level() -> int:
    """
    :return: The minimum level that wild encounters can have, given the current Repel
             state and the level of the first non-fainted Pokémon.
    """
    first_non_fainted_party_member = get_party().first_non_fainted
    if first_non_fainted_party_member is None:
        return 0
    else:
        return get_party().first_non_fainted.level if get_event_var("REPEL_STEP_COUNT") > 0 else 0
