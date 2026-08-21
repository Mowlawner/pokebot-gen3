"""Opt-in, append-only diagnostics for the disposable Route 102 audit."""

import json
import os
from pathlib import Path
from threading import Lock

_lock = Lock()
_sequence = 0


def emit(event: str, **fields) -> None:
    """Append one durable record when ROUTE102_DIAGNOSTIC_PATH is configured."""
    global _sequence
    output = os.environ.get("ROUTE102_DIAGNOSTIC_PATH")
    if not output:
        return
    with _lock:
        _sequence += 1
        record = {"marker": "ROUTE102_LOOP_DIAGNOSTIC", "sequence": _sequence, "event": event, **fields}
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True, default=str) + "\n")
            stream.flush()
        print(f"ROUTE102_LOOP_DIAGNOSTIC seq={_sequence} event={event}", flush=True)
