"""Opt-in periodic Python stack dumps for diagnosing silent stalls.

A run that stops logging while the emulator keeps running gives no hint of
where the bot is stuck, and the normal log only writes at decision boundaries.
Setting ``POKEBOT_STALL_DUMP`` to a file path makes a watchdog thread write every
other thread's Python stack to that file on a fixed interval, independent of the
log.  ``POKEBOT_STALL_DUMP_SECONDS`` sets the interval (default 60).  Disabled
by default and has no effect on normal runs.

The dump is taken from ordinary Python (``sys._current_frames``) and not from
``faulthandler``.  ``faulthandler`` reads other threads' frames without the
interpreter lock, and a live Windows run died silently, mid-dump, while the main
thread was inside a native emulator call.  The watchdog needs the interpreter
lock to dump, so it cannot show a thread stuck in native code that never
releases it; it does show any Python-level stall, which is what this tool is for.
"""

from __future__ import annotations

import os
import sys
import threading
import traceback
from datetime import datetime
from typing import Mapping, TextIO

DEFAULT_INTERVAL_SECONDS = 60.0

_dump_file: TextIO | None = None
_thread: threading.Thread | None = None
_stop = threading.Event()


def _parse_interval(environ: Mapping[str, str]) -> float:
    try:
        interval = float(environ.get("POKEBOT_STALL_DUMP_SECONDS", DEFAULT_INTERVAL_SECONDS))
    except ValueError:
        return DEFAULT_INTERVAL_SECONDS
    return interval if interval > 0 else DEFAULT_INTERVAL_SECONDS


def write_stacks(file: TextIO) -> None:
    """Write a timestamped stack for every thread except the caller."""

    names = {thread.ident: thread.name for thread in threading.enumerate()}
    caller = threading.get_ident()
    file.write(f"=== stack dump {datetime.now().isoformat(timespec='seconds')} ===\n")
    for ident, frame in sorted(sys._current_frames().items()):
        if ident == caller:
            continue
        file.write(f"Thread {names.get(ident, '?')} (0x{ident:x}), oldest call first:\n")
        file.write("".join(traceback.format_stack(frame)))
        file.write("\n")
    file.flush()


def _watch(file: TextIO, interval: float, stop: threading.Event) -> None:
    while not stop.wait(interval):
        try:
            write_stacks(file)
        except (OSError, ValueError):
            return


def enable_from_environment(environ: Mapping[str, str] | None = None) -> str | None:
    """Start periodic stack dumps when requested; return the dump path or None."""

    global _dump_file, _thread
    environ = os.environ if environ is None else environ
    path = environ.get("POKEBOT_STALL_DUMP")
    if not path:
        return None
    disable()
    try:
        _dump_file = open(path, "a", buffering=1, encoding="utf-8", errors="replace")
    except OSError:
        return None
    _stop.clear()
    _thread = threading.Thread(
        target=_watch,
        args=(_dump_file, _parse_interval(environ), _stop),
        name="pokebot-stall-dump",
        daemon=True,
    )
    _thread.start()
    return path


def disable() -> None:
    """Stop periodic dumps and close the dump file."""

    global _dump_file, _thread
    _stop.set()
    if _thread is not None:
        _thread.join(timeout=5)
        _thread = None
    if _dump_file is not None:
        _dump_file.close()
        _dump_file = None
