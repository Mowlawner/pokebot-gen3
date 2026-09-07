"""Opt-in rolling trace for correlating visible emulator stalls with bot work."""

from collections import deque
from contextlib import contextmanager
import json
from pathlib import Path
from time import perf_counter_ns
from functools import wraps
import threading


_background_state = threading.local()


def background_work_suppressed() -> bool:
    """Return whether the current thread is running non-frame work.

    Background route planning shares the process with the emulator's frame
    loop.  It must not append spans, marks, or durations to the mutable trace
    object owned by that loop.
    """

    return bool(getattr(_background_state, "suppressed", False))


@contextmanager
def suppress_background_instrumentation():
    """Disable frame instrumentation for the current worker thread only."""

    previous = background_work_suppressed()
    _background_state.suppressed = True
    try:
        yield
    finally:
        _background_state.suppressed = previous


class StutterTrace:
    """Keep only triggered frame windows; no work is done while disabled."""

    def __init__(self, enabled: bool = False, threshold_ms: float = 50.0, output: Path | None = None):
        self.enabled = enabled
        self.threshold_ns = int(threshold_ms * 1_000_000)
        self.output = output
        self._frames: deque[dict] = deque(maxlen=35)
        self._region: list[dict] | None = None
        self._after = 0
        self._last_trigger_frame: int | None = None
        self.current: dict | None = None
        self._span_stack: list[dict] = []

    def begin(
        self, application_frame: int, emulator_frame: int, cached_active: bool = False, actions_remaining: int = 0
    ) -> None:
        if not self.enabled:
            return
        self.current = {
            "application_frame": application_frame,
            "emulator_frame": emulator_frame,
            "map_id": None,
            "player_position": None,
            "cached_route_active": cached_active,
            "cached_route_actions_remaining": actions_remaining,
            "fast_path_used": False,
            "fast_path_duration_ms": 0.0,
            "route_invalidated": False,
            "invalidation_reason": None,
            "state_transition": None,
            "full_perception_ran": False,
            "perception_duration_ms": 0.0,
            "goal_evaluation_ran": False,
            "goal_evaluation_duration_ms": 0.0,
            "pathfinding_ran": False,
            "pathfinding_duration_ms": 0.0,
            "controller_duration_ms": 0.0,
            "emulator_advancement_duration_ms": 0.0,
            "emulator_core_run_duration_ms": 0.0,
            "emulator_frame_callback_duration_ms": 0.0,
            "emulator_throttle_duration_ms": 0.0,
            "nuzlocke_game_state_duration_ms": 0.0,
            "nuzlocke_duration_ms": 0.0,
            "total_frame_duration_ms": 0.0,
            "map_warp_transition": False,
            "storage_raw_read_duration_ms": 0.0,
            "storage_raw_compare_duration_ms": 0.0,
            "storage_box_parse_duration_ms": 0.0,
            "storage_snapshot_duration_ms": 0.0,
            "nuzlocke_storage_access_duration_ms": 0.0,
            "storage_snapshot_cache_hit": False,
            "storage_slots_parsed": 0,
            "route_invalidation_detail": None,
            "nuzlocke_party_duration_ms": 0.0,
            "nuzlocke_bag_duration_ms": 0.0,
            "nuzlocke_player_duration_ms": 0.0,
            "nuzlocke_battle_duration_ms": 0.0,
            "nuzlocke_progression_duration_ms": 0.0,
            "nuzlocke_observer_duration_ms": 0.0,
            "nuzlocke_storage_event_index_duration_ms": 0.0,
            "emulator_advance_start_wall_ns": None,
            "emulator_advance_end_wall_ns": None,
            "inter_frame_interval_ms": None,
            "map_changed_event": False,
            "route_replanned": False,
            "gui_frame_presented": False,
            "overworld_map_metadata_lookup_duration_ms": 0.0,
            "overworld_static_tile_preparation_duration_ms": 0.0,
            "overworld_runtime_object_scan_duration_ms": 0.0,
            "overworld_trigger_binding_duration_ms": 0.0,
            "navigation_world_construction_duration_ms": 0.0,
            "world_map_graph_lookup_duration_ms": 0.0,
            "world_map_route_lookup_duration_ms": 0.0,
            "navigation_planning_duration_ms": 0.0,
            "warp_destination_prewarm_duration_ms": 0.0,
            "warp_destination_prewarm_used": False,
            "warp_destination_prewarm_failed": False,
            "timing_tree": [],
        }
        self._span_stack = []

    @contextmanager
    def span(self, name: str):
        """Record an inclusive/exclusive wall-clock span for the current frame.

        This is deliberately tied to the targeted stutter trace rather than
        the aggregate profiler. It records only when the trace is enabled and
        does not inspect, cache, or alter navigation state.
        """
        if background_work_suppressed() or not self.enabled or self.current is None:
            yield
            return
        started = perf_counter_ns()
        node = {"name": name, "inclusive_ms": 0.0, "exclusive_ms": 0.0, "calls": 1, "children": []}
        parent = self._span_stack[-1] if self._span_stack else None
        if parent is None:
            self.current["timing_tree"].append(node)
        else:
            parent["children"].append(node)
        self._span_stack.append(node)
        try:
            yield
        finally:
            elapsed = perf_counter_ns() - started
            node["inclusive_ms"] = round(elapsed / 1_000_000, 3)
            node["exclusive_ms"] = round(
                (elapsed - sum(child["inclusive_ms"] for child in node["children"]) * 1_000_000) / 1_000_000,
                3,
            )
            self._span_stack.pop()

    def mark(self, name: str, value=True) -> None:
        if not background_work_suppressed() and self.enabled and self.current is not None:
            self.current[name] = value

    def duration(self, name: str, started_ns: int) -> None:
        if not background_work_suppressed() and self.enabled and self.current is not None:
            self.current[name] = round((perf_counter_ns() - started_ns) / 1_000_000, 3)

    def now(self) -> int:
        return perf_counter_ns() if self.enabled and not background_work_suppressed() else 0

    def call(self, name: str, function, *args, **kwargs):
        """Call one operation under a named span without changing its result."""
        if background_work_suppressed():
            return function(*args, **kwargs)
        with self.span(name):
            return function(*args, **kwargs)

    def finish(self) -> None:
        if background_work_suppressed() or not self.enabled or self.current is None:
            return
        frame = self.current
        self.current = None
        self._frames.append(frame)
        triggered = (
            frame["total_frame_duration_ms"] * 1_000_000 > self.threshold_ns
            or frame["route_invalidated"]
            or frame["goal_evaluation_ran"]
            and frame["cached_route_active"]
            or frame["pathfinding_ran"]
            and frame["cached_route_active"]
            or frame["map_warp_transition"]
        )
        if triggered:
            if (
                self._region is None
                or self._last_trigger_frame is None
                or frame["application_frame"] > self._last_trigger_frame + 1
            ):
                self._flush()
                self._region = list(self._frames)
            elif self._region[-1]["application_frame"] != frame["application_frame"]:
                self._region.append(frame)
            self._last_trigger_frame = frame["application_frame"]
            self._after = 30
        elif self._region is not None and self._after > 0:
            self._region.append(frame)
            self._after -= 1
            if self._after == 0:
                self._flush()

    def _flush(self) -> None:
        if not self._region:
            return
        payload = {
            "trigger_frames": [
                f["application_frame"]
                for f in self._region
                if f["route_invalidated"]
                or f["total_frame_duration_ms"] > self.threshold_ns / 1_000_000
                or f["map_warp_transition"]
            ],
            "frames": self._region,
        }
        if self.output is not None:
            self.output.parent.mkdir(parents=True, exist_ok=True)
            with self.output.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(payload, sort_keys=True) + "\n")
        self._region = None

    def close(self) -> None:
        if self.enabled:
            self._flush()


def traced(name: str):
    """Decorate a concrete operation for the targeted per-frame call tree."""

    def decorator(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            from modules.context import context

            trace = getattr(context, "stutter_trace", None)
            if trace is None:
                return function(*args, **kwargs)
            with trace.span(name):
                return function(*args, **kwargs)

        return wrapped

    return decorator
