"""Diagnostic-only direct mGBA replay of the mart freeze save state."""

from pathlib import Path
import sys
import time
import zlib

import mgba.core
import mgba.image
import mgba.vfs


def savestate_bytes(path: Path) -> bytes:
    data = path.read_bytes()
    offset = 8
    while offset + 12 <= len(data):
        length = int.from_bytes(data[offset : offset + 4], "big")
        chunk_type = data[offset + 4 : offset + 8]
        chunk = data[offset + 8 : offset + 8 + length]
        if chunk_type == b"gbAs":
            return zlib.decompress(chunk)
        offset += 12 + length
    raise RuntimeError(f"No gbAs mGBA state chunk found in {path}")


def main() -> None:
    profile = Path(sys.argv[1])
    rom_path = Path("roms") / "Pokemon - Emerald Version (USA, Europe).gba"
    state_path = profile / "current_state.ss1"
    state = savestate_bytes(state_path)

    core = mgba.core.load_path(str(rom_path))
    if core is None:
        raise RuntimeError(f"Could not load ROM: {rom_path}")
    save = mgba.vfs.open_path(str(profile / "current_save.sav"), "r+")
    core.load_save(save)
    screen = mgba.image.Image(*core.desired_video_dimensions())
    core.set_video_buffer(screen)
    core.reset()
    state_file = mgba.vfs.VFile.fromEmpty()
    state_file.write(state, len(state))
    state_file.seek(0, whence=0)
    core.load_state(state_file)

    print(f"loaded_frame={core.frame_counter}", flush=True)
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    press_at = int(sys.argv[3]) if len(sys.argv) > 3 else None
    for index in range(limit):
        before = core.frame_counter
        started = time.monotonic()
        print(f"run_begin index={index} frame={before}", flush=True)
        core.set_keys(raw=1 if press_at == index else 0)
        core.run_frame()
        print(
            f"run_end index={index} frame={core.frame_counter} elapsed={time.monotonic() - started:.3f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
