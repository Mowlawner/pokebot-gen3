"""Low-overhead, opt-in instrumentation for the live control loop."""

from collections import Counter
from functools import wraps
from time import perf_counter_ns

_timings: Counter[str] = Counter()
_counts: Counter[str] = Counter()
_invalidation_reasons: Counter[str] = Counter()


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
        _timings[name] += perf_counter_ns() - started_at


def profiled(timing_name: str, counter_name: str):
    """Decorate a coarse boundary without timing it when profiling is off."""

    def decorator(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            if not enabled():
                return function(*args, **kwargs)
            count(counter_name)
            started_at = now()
            try:
                return function(*args, **kwargs)
            finally:
                timing(timing_name, started_at)

        return wrapped

    return decorator


def invalidation(reason: str) -> None:
    if enabled():
        _counts["cached_plan_invalidations"] += 1
        _invalidation_reasons[reason] += 1


def snapshot() -> dict[str, int | dict[str, int]]:
    """Return a cheap, serializable cumulative snapshot for profile output."""
    if not enabled():
        return {}
    return {
        **{f"{name}_ms": round(value / 1_000_000, 3) for name, value in _timings.items()},
        **dict(_counts),
        "cached_plan_invalidation_reasons": dict(_invalidation_reasons),
    }


def format_snapshot() -> str:
    values = snapshot()
    if not values:
        return ""
    fields = " ".join(f"{name}={value!r}" for name, value in sorted(values.items()))
    return f"PROFILE: {fields}"
