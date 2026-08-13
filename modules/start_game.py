"""Pure helpers for resolving fresh-game player initialization values."""

from dataclasses import dataclass
from enum import Enum
import random
from typing import Protocol

from modules.keyboard import get_current_keyboard_layout


# Western Gen III player names contain at most seven visible characters. The
# terminator in SaveBlock2 is storage, not part of this limit.
PLAYER_NAME_MAX_LENGTH = 7


class RandomSource(Protocol):
    def randint(self, start: int, end: int) -> int: ...

    def choice(self, sequence): ...


class PlayerGender(Enum):
    MALE = "male"
    FEMALE = "female"


class PlayerGenderPolicy(Enum):
    RANDOM = "random"
    EXPLICIT = "explicit"


class PlayerNamePolicy(Enum):
    HUMAN_RANDOM = "random"
    GIBBERISH = "gibberish"
    EXPLICIT = "explicit"


@dataclass(frozen=True)
class StartGameInitialization:
    """Concrete values resolved once for one Start New Game run."""

    gender: PlayerGender
    name: str


def player_gender_policy(value: str) -> PlayerGenderPolicy:
    return PlayerGenderPolicy.RANDOM if value.lower() == "random" else PlayerGenderPolicy.EXPLICIT


def player_name_policy(value: str) -> PlayerNamePolicy:
    normalized = value.lower()
    if normalized == "random":
        return PlayerNamePolicy.HUMAN_RANDOM
    if normalized == "gibberish":
        return PlayerNamePolicy.GIBBERISH
    return PlayerNamePolicy.EXPLICIT


def _random_name_characters() -> tuple[str, ...]:
    """Return sensible name characters from the active game's keyboard."""
    characters = get_current_keyboard_layout().valid_characters
    return tuple(dict.fromkeys(character for character in characters if character.isalpha()))


def generate_gibberish_player_name(
    rng: RandomSource | None = None,
    *,
    max_length: int = PLAYER_NAME_MAX_LENGTH,
) -> str:
    """Generate a valid-character string for the deliberate gibberish option."""
    if max_length < 1:
        raise ValueError("max_length must be positive")
    if rng is None:
        rng = random.Random()
    characters = _random_name_characters()
    if not characters:
        raise ValueError("The active keyboard has no alphabetic name characters")
    length = rng.randint(1, max_length)
    return "".join(rng.choice(characters) for _ in range(length))


# Preserve the original helper name for callers that used the old random-string
# behavior before the human-name option was introduced.
generate_random_player_name = generate_gibberish_player_name


def _get_human_first_name(gender: PlayerGender) -> str:
    """Isolate the third-party name source for testing and future replacement."""
    import names

    return names.get_first_name(gender=gender.value)


def generate_human_player_name(
    gender: PlayerGender,
    *,
    max_length: int = PLAYER_NAME_MAX_LENGTH,
) -> str:
    """Request human first names until one fits; never truncate a name."""
    if max_length < 1:
        raise ValueError("max_length must be positive")
    while True:
        name = _get_human_first_name(gender)
        if 0 < len(name) <= max_length:
            return name


def _resolve_gender(configured_gender: str, rng: RandomSource) -> PlayerGender:
    if player_gender_policy(configured_gender) is PlayerGenderPolicy.RANDOM:
        return rng.choice((PlayerGender.MALE, PlayerGender.FEMALE))
    try:
        return PlayerGender(configured_gender.lower())
    except ValueError as error:
        raise ValueError(f"Invalid player gender: {configured_gender!r}") from error


def _validate_explicit_name(name: str, max_length: int) -> str:
    if not name or len(name) > max_length:
        raise ValueError(f"Player names must contain between 1 and {max_length} characters")
    return name


def resolve_start_game_initialization(
    configured_name: str,
    configured_gender: str,
    rng: RandomSource | None = None,
    *,
    max_length: int = PLAYER_NAME_MAX_LENGTH,
) -> StartGameInitialization:
    """Resolve concrete gender and name values exactly once for a new game."""
    if rng is None:
        rng = random.Random()
    gender = _resolve_gender(configured_gender, rng)
    name_policy = player_name_policy(configured_name)
    if name_policy is PlayerNamePolicy.EXPLICIT:
        name = _validate_explicit_name(configured_name, max_length)
    elif name_policy is PlayerNamePolicy.GIBBERISH:
        name = generate_gibberish_player_name(rng, max_length=max_length)
    else:
        name = generate_human_player_name(gender, max_length=max_length)
    return StartGameInitialization(gender, name)


# Compatibility wrapper for callers that only need a configured name.
def resolve_player_name(
    configured_name: str,
    rng: RandomSource | None = None,
    *,
    max_length: int = PLAYER_NAME_MAX_LENGTH,
) -> str:
    return resolve_start_game_initialization(configured_name, "random", rng, max_length=max_length).name
