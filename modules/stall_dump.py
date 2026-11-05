"""Opt-in periodic Python stack dumps for diagnosing silent stalls.

A run that stops logging while the emulator keeps running gives no hint of
where the bot is stuck, and the normal log only writes at decision boundaries.
Setting ``POKEBOT_STALL_DUMP`` to a file path makes the interpreter write every
thread's Python stack to that file on a fixed interval, independent of the
log.  ``POKEBOT_STALL_DUMP_SECONDS`` sets the interval (default 60).  Disabled
by default and has no effect on normal runs.
"""

from __future__ import annotations

import faulthandler
import os
from typing import Mapping

DEFAULT_INTERVAL_SECONDS = 60.0

_dump_file = None


def enable_from_environment(environ: Mapping[str, str] | None = None) -> str | None:
    """Start periodic stack dumps when requested; return the dump path or None."""

    global _dump_file
    environ = os.environ if environ is None else environ
    path = environ.get("POKEBOT_STALL_DUMP")
    if not path:
        return None
    try:
        interval = float(environ.get("POKEBOT_STALL_DUMP_SECONDS", DEFAULT_INTERVAL_SECONDS))
    except ValueError:
        interval = DEFAULT_INTERVAL_SECONDS
    if interval <= 0:
        interval = DEFAULT_INTERVAL_SECONDS
    try:
        _dump_file = open(path, "a", buffering=1)
        faulthandler.dump_traceback_later(interval, repeat=True, file=_dump_file)
    except OSError:
        return None
    return path


def disable() -> None:
    """Stop periodic dumps and close the dump file."""

    global _dump_file
    faulthandler.cancel_dump_traceback_later()
    if _dump_file is not None:
        _dump_file.close()
        _dump_file = None
