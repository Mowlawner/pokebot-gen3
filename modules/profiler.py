"""Low-overhead, opt-in instrumentation for the live control loop."""

from collections import Counter
from functools import wraps
from time import perf_counter_ns

_timings: Counter[str] = Counter()
_timing_calls: Counter[str] = Counter()
_timing_max: dict[str, int] = {}
_counts: Counter[str] = Counter()
_invalidation_reasons: Counter[str] = Counter()
_output_call_count = 0
_pathfinding_searches: list[dict[str, object]] = []
_cross_map_goal_evaluations: list[dict[str, object]] = []


def enabled() -> bool:
    """Return whether performance instrumentation is currently enabled."""
    from modules.context import context

    return bool(getattr(context, "debug_profile", False))


def now() -> int:
    """Read a timer only for an enabled probe."""
    return perf_counter_ns() if enabled() else 0


def count(name: str, amount: int = 1) -> None:
    if enabled():
        _counts[name] += amount


def timing(name: str, started_at: int) -> None:
    if enabled():
        elapsed = perf_counter_ns() - started_at
        _timings[name] += elapsed
        _timing_calls[name] += 1
        if elapsed > _timing_max.get(name, 0):
            _timing_max[name] = elapsed


def profiled(timing_name: str, counter_name: str):
    """Decorate a coarse boundary without timing it when profiling is off."""

    def decorator(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            from modules.context import context

            trace = getattr(context, "stutter_trace", None)
            if not enabled() and (trace is None or not trace.enabled):
                return function(*args, **kwargs)
            trace_start = trace.now() if trace is not None else 0
            if enabled():
                count(counter_name)
            started_at = now() if enabled() else 0
            try:
                return function(*args, **kwargs)
            finally:
                if enabled():
                    timing(timing_name, started_at)
                if trace is not None:
                    fields = {
                        "cached_route_execution": ("fast_path_duration_ms", "fast_path_used"),
                        "agent_goal_evaluation": ("goal_evaluation_duration_ms", "goal_evaluation_ran"),
                        "navigation_pathfinding": ("pathfinding_duration_ms", "pathfinding_ran"),
                    }
                    field = fields.get(timing_name)
                    if field is not None:
                        trace.duration(field[0], trace_start)
                        if timing_name != "cached_route_execution":
                            trace.mark(field[1], True)

        return wrapped

    return decorator


def invalidation(reason: str) -> None:
    if enabled():
        _counts["cached_plan_invalidations"] += 1
        _invalidation_reasons[reason] += 1


def record_pathfinding_search(**fields: object) -> dict[str, object] | None:
    """Store one detailed search record when profiling is explicitly enabled."""
    if enabled():
        _pathfinding_searches.append(fields)
        return fields
    return None


def pathfinding_searches() -> tuple[dict[str, object], ...]:
    """Return detailed pathfinding records for benchmark/debug consumers."""
    return tuple(_pathfinding_searches) if enabled() else ()


def clear_pathfinding_searches() -> None:
    if enabled():
        _pathfinding_searches.clear()


def record_cross_map_goal_evaluation(**fields: object) -> dict[str, object] | None:
    """Store one opt-in cross-map candidate audit record."""
    if enabled():
        _cross_map_goal_evaluations.append(fields)
        return fields
    return None


def cross_map_goal_evaluations() -> tuple[dict[str, object], ...]:
    """Return cross-map candidate records for benchmark/debug consumers."""
    return tuple(_cross_map_goal_evaluations) if enabled() else ()


def clear_cross_map_goal_evaluations() -> None:
    if enabled():
        _cross_map_goal_evaluations.clear()


def should_emit_output(every: int = 1) -> bool:
    """Throttle expensive formatted profile output while retaining counters."""
    global _output_call_count
    if every <= 1:
        return True
    _output_call_count += 1
    return _output_call_count % every == 0


def snapshot() -> dict[str, int | dict[str, int]]:
    """Return a cheap, serializable cumulative snapshot for profile output."""
    if not enabled():
        return {}
    return {
        **{f"{name}_ms": round(value / 1_000_000, 3) for name, value in _timings.items()},
        **{f"{name}_calls": value for name, value in _timing_calls.items()},
        **{f"{name}_avg_ms": round(_timings[name] / value / 1_000_000, 3) for name, value in _timing_calls.items()},
        **{f"{name}_max_ms": round(_timing_max[name] / 1_000_000, 3) for name in _timing_calls},
        **dict(_counts),
        "cached_plan_invalidation_reasons": dict(_invalidation_reasons),
    }


def format_snapshot() -> str:
    values = snapshot()
    if not values:
        return ""
    fields = " ".join(f"{name}={value!r}" for name, value in sorted(values.items()))
    return f"PROFILE: {fields}"
