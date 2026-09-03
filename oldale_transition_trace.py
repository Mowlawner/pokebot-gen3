"""Temporary diagnostic for the Oldale Town -> Oldale Mart boundary."""

import faulthandler
import runpy
import sys
import time

from mgba.core import Core

_original_core_run_frame = Core.run_frame
_trace_until_application_frame = None
_trace_count = 0
_last_reported_location = None


def _safe_location(emulator):
    try:
        from modules.player import get_player_avatar

        avatar = get_player_avatar()
        return (
            getattr(avatar, "map_group_and_number", None),
            getattr(avatar, "local_coordinates", None),
        )
    except Exception as error:  # diagnostic-only fallback
        return ("<unavailable>", repr(error))


def _trace_enabled(application_frame, location):
    global _trace_until_application_frame
    map_id = location[0]
    if map_id == (0, 10):
        _trace_until_application_frame = application_frame + 300
    return _trace_until_application_frame is not None and application_frame <= _trace_until_application_frame


def _core_run_frame_with_trace(self):
    try:
        from modules.context import context

        application_frame = getattr(context, "frame", None)
    except Exception:
        application_frame = None
    enabled = _trace_until_application_frame is not None and (
        application_frame is None or application_frame <= _trace_until_application_frame
    )
    if enabled:
        print(
            f"OLDALE_CORE_BEGIN app={application_frame!r} emulator={self.frame_counter!r}",
            flush=True,
        )
    result = _original_core_run_frame(self)
    if enabled:
        print(
            f"OLDALE_CORE_END app={application_frame!r} emulator={self.frame_counter!r}",
            flush=True,
        )
    return result


Core.run_frame = _core_run_frame_with_trace


def _install_emulator_boundary_trace():
    from modules.context import context
    from modules.libmgba import LibmgbaEmulator

    original_run_single_frame = LibmgbaEmulator.run_single_frame

    def run_single_frame_with_trace(self):
        global _trace_count, _last_reported_location
        application_frame = getattr(context, "frame", None)
        location_before = _safe_location(self)
        enabled = isinstance(application_frame, int) and _trace_enabled(application_frame, location_before)
        started = time.monotonic()
        if location_before != _last_reported_location:
            print(
                f"MAP_CHANGE_BEFORE app={application_frame!r} emulator={self.get_frame_count()!r} "
                f"location={location_before!r}",
                flush=True,
            )
            _last_reported_location = location_before
        if enabled:
            _trace_count += 1
            print(
                f"OLDALE_APP_BEGIN count={_trace_count} app={application_frame!r} "
                f"emulator={self.get_frame_count()!r} location={location_before!r}",
                flush=True,
            )
        try:
            return original_run_single_frame(self)
        finally:
            location_after = _safe_location(self)
            if location_after != _last_reported_location:
                print(
                    f"MAP_CHANGE_AFTER app={application_frame!r} emulator={self.get_frame_count()!r} "
                    f"location={location_after!r} elapsed={time.monotonic() - started:.3f}",
                    flush=True,
                )
                _last_reported_location = location_after
            if enabled:
                print(
                    f"OLDALE_APP_END count={_trace_count} app={application_frame!r} "
                    f"emulator={self.get_frame_count()!r} location={_safe_location(self)!r} "
                    f"elapsed={time.monotonic() - started:.3f}",
                    flush=True,
                )

    LibmgbaEmulator.run_single_frame = run_single_frame_with_trace


faulthandler.enable()
faulthandler.dump_traceback_later(3, repeat=True)
_install_emulator_boundary_trace()
sys.argv = ["pokebot.py", *sys.argv[1:]]
runpy.run_path("pokebot.py", run_name="__main__")
