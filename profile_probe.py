"""Diagnostic-only profile state probe."""

import sys
from pathlib import Path

from modules.context import context
from modules.game import set_rom
from modules.libmgba import LibmgbaEmulator
from modules.memory import get_game_state
from modules.player import get_player_avatar
from modules.profiles import load_profile_by_name
from modules.tasks import get_global_script_context, get_tasks


def main() -> None:
    names = sys.argv[1:]
    if names == ["all"]:
        names = sorted(
            path.name
            for path in Path("profiles").iterdir()
            if path.is_dir() and (path / "metadata.yml").is_file() and (path / "current_state.ss1").is_file()
        )
    for name in names:
        profile = load_profile_by_name(name)
        context.profile = profile
        set_rom(profile.rom)
        context.emulator = LibmgbaEmulator(profile, lambda: None, save_state_on_shutdown=False)
        avatar = get_player_avatar()
        script = get_global_script_context()
        print(
            {
                "profile": name,
                "frame": context.emulator.get_frame_count(),
                "game_state": get_game_state(),
                "map": getattr(avatar, "map_group_and_number", None),
                "coordinates": getattr(avatar, "local_coordinates", None),
                "facing": getattr(avatar, "facing", None),
                "script": getattr(script, "script_function_name", None),
                "script_active": getattr(script, "is_active", None),
                "tasks": tuple(task.symbol for task in (get_tasks() or ())),
            },
            flush=True,
        )
        context.emulator = None


if __name__ == "__main__":
    main()
