import contextlib
import errno
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from modules import exceptions
from modules.config import load_config_file, save_config_file
from modules.config.schemas_v1 import ProfileMetadata, ProfileMetadataROM
from modules.console import console
from modules.roms import ROM, ROMS_DIRECTORY, list_available_roms, load_rom_data
from modules.runtime import get_base_path

PROFILES_DIRECTORY = get_base_path() / "profiles"


@dataclass
class Profile:
    """
    Profiles are config directories that contain all data for a save game, such as saves,
    screenshots, stats, custom config, etc. -- except the ROM itself.

    The only requirements for a Profile are: There must be a subdirectory in `profiles/` with
    the name of the profile, and inside that directory there must be a file called
    `metadata.yml` that satisfies the `metadata_schema`.

    This metadata file specifies which game/ROM a save game is associated with, so we can
    load the correct ROM when selected.
    """

    rom: ROM
    path: Path
    last_played: datetime | None


class ProfileInUseError(RuntimeError):
    """The selected profile is already owned by another bot process."""


class ProfileLock:
    """Own one profile for the lifetime of an emulator process.

    The lock file is intentionally retained as a harmless marker.  The OS
    lock is held on its descriptor, so a crashed process cannot leave a stale
    lock that prevents a later run from starting.  This protects both the
    save-state and the durable event log, which must always describe one
    emulator timeline.
    """

    filename = ".pokebot.lock"

    def __init__(self, profile_path: str | Path):
        self.profile_path = Path(profile_path)
        self.path = self.profile_path / self.filename
        self._descriptor: int | None = None

    @property
    def acquired(self) -> bool:
        return self._descriptor is not None

    def acquire(self) -> "ProfileLock":
        if self.acquired:
            return self
        self.profile_path.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(str(self.path), os.O_RDWR | os.O_CREAT, 0o600)
        try:
            self._lock_descriptor(descriptor)
            os.ftruncate(descriptor, 0)
            os.write(descriptor, f"pid={os.getpid()}\n".encode("ascii"))
            os.fsync(descriptor)
        except (BlockingIOError, OSError) as error:
            try:
                os.close(descriptor)
            except OSError:
                pass
            if isinstance(error, BlockingIOError) or getattr(error, "errno", None) in (errno.EACCES, errno.EAGAIN):
                raise ProfileInUseError(
                    f"Profile {self.profile_path.name!r} is already in use by another bot process."
                ) from error
            raise RuntimeError(f"Could not lock profile {self.profile_path}: {error}") from error
        self._descriptor = descriptor
        return self

    @staticmethod
    def _lock_descriptor(descriptor: int) -> None:
        if os.name == "nt":
            # msvcrt.locking locks bytes rather than descriptors. Ensure the
            # first byte exists and position the descriptor at it.
            import msvcrt

            if os.fstat(descriptor).st_size == 0:
                os.write(descriptor, b"\0")
            os.lseek(descriptor, 0, os.SEEK_SET)
            msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
            return
        import fcntl

        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def release(self) -> None:
        descriptor, self._descriptor = self._descriptor, None
        if descriptor is None:
            return
        try:
            if os.name == "nt":
                import msvcrt

                os.lseek(descriptor, 0, os.SEEK_SET)
                msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)

    def __enter__(self) -> "ProfileLock":
        return self.acquire()

    def __exit__(self, exception_type, exception, traceback) -> None:
        self.release()


def list_available_profiles() -> list[Profile]:
    if not PROFILES_DIRECTORY.is_dir():
        raise RuntimeError(f"Directory {str(PROFILES_DIRECTORY)} does not exist!")

    profiles = []
    for entry in PROFILES_DIRECTORY.iterdir():
        if entry.name.startswith("_"):
            continue
        try:
            profile = load_profile(entry)
            profiles.append(profile)
        except RuntimeError:
            pass
    return profiles


def load_profile_by_name(name: str) -> Profile:
    return load_profile(PROFILES_DIRECTORY / name)


def load_profile(path: Path) -> Profile:
    if not path.is_dir():
        raise RuntimeError("Path is not a valid profile directory.")
    metadata = load_config_file(path / ProfileMetadata.filename, ProfileMetadata, strict=True)
    current_state = path / "current_state.ss1"
    if current_state.exists():
        last_played = datetime.fromtimestamp(current_state.stat().st_mtime)
    else:
        last_played = None

    rom_file = ROMS_DIRECTORY / metadata.rom.file_name
    if rom_file.is_file():
        rom = load_rom_data(rom_file)
        if rom.is_gen2:
            raise RuntimeError("Only Generation 3 games are supported")
        return Profile(rom, path, last_played)
    else:
        for rom in list_available_roms():
            if all(
                [
                    rom.game_code == metadata.rom.game_code,
                    rom.revision == metadata.rom.revision,
                    rom.language.value == metadata.rom.language,
                ]
            ):
                return Profile(rom, path, last_played)

    console.print(
        f"[bold red]Could not find ROM `{metadata.rom.file_name}` for profile `{path.name}`, "
        f"please place `{metadata.rom.file_name}` into `{ROMS_DIRECTORY}`!"
    )
    sys.exit(1)


def profile_directory_exists(name: str) -> bool:
    return (PROFILES_DIRECTORY / name).is_dir()


def create_profile(name: str, rom: ROM) -> Profile:
    if name.startswith("_"):
        raise exceptions.PrettyValueError('Profile names cannot start with the underscore "_" character.')
    profile_directory = PROFILES_DIRECTORY / name
    if profile_directory.exists():
        raise RuntimeError(f'There already is a profile called "{name}", cannot create a new one with that name.')

    rom_cfg = ProfileMetadataROM(
        file_name=rom.file.name,
        game_code=rom.game_code,
        revision=rom.revision,
        language=str(rom.language),
    )
    profile_metadata = ProfileMetadata(rom=rom_cfg)
    save_config_file(profile_directory, profile_metadata, strict=False)

    return Profile(rom, profile_directory, None)
