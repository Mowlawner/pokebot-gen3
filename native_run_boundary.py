"""Diagnostic-only before/after boundary trace for the normal mGBA call."""

import runpy
import sys
import faulthandler
from mgba.core import Core

faulthandler.dump_traceback_later(2, repeat=True)


original_run_frame = Core.run_frame


def run_frame_with_boundary_trace(self):
    frame = self.frame_counter
    if 413960 <= frame <= 413990:
        try:
            keys = self._core.getKeys(self._core)
        except Exception:
            keys = None
        cpu = getattr(self, "cpu", None)
        print(
            f"BOUNDARY_BEGIN frame={frame} keys={keys!r} "
            f"pc={getattr(cpu, 'pc', None)!r} "
            f"keys_active={getattr(getattr(self, '_native', None), 'keysActive', None)!r} "
            f"keys_last={getattr(getattr(self, '_native', None), 'keysLast', None)!r}",
            flush=True,
        )
    result = original_run_frame(self)
    if 413960 <= frame <= 413990:
        print(f"BOUNDARY_END frame={self.frame_counter}", flush=True)
    return result


Core.run_frame = run_frame_with_boundary_trace
sys.argv = ["pokebot.py", *sys.argv[1:]]
runpy.run_path("pokebot.py", run_name="__main__")
