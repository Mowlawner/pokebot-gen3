"""Disposable live validation mode for one explicitly armed gate edge."""

from typing import Generator

from modules.agent_control import AgentControlLoop, NavigationError, observe_agent
from modules.console import diagnostic_print
from modules.goals import GoalConstraints, NavigationGoal, ReachWarp
from modules.memory import GameState, get_game_state, pack_uint16, read_symbol, set_event_var, write_symbol
from modules.map import get_runtime_object_table
from modules.navigation import NavigationAction, NavigationActionType
from modules.overworld import perceive_overworld
from modules.state_cache import state_cache

from ._interface import BotMode


def _fixture_map_identity(map_id):
    value = getattr(map_id, "value", map_id)
    return tuple(value) if isinstance(value, (list, tuple)) else value


def _write_object_coordinates(slot: int, coordinates: tuple[int, int]) -> None:
    encoded = pack_uint16(coordinates[0] + 7) + pack_uint16(coordinates[1] + 7)
    base = slot * 0x24
    for offset in (0x0C, 0x10, 0x14):
        write_symbol("gObjectEvents", encoded, offset=base + offset)


def _apply_gate_fixture(fixture, *, current_map, expected_map) -> None:
    """Apply an explicitly configured, disposable source-derived fixture."""

    fixture_map = fixture.source_map or _fixture_map_identity(expected_map)
    if _fixture_map_identity(current_map) != fixture_map:
        raise ValueError(
            f"gate fixture expects map {fixture_map!r}, observed {_fixture_map_identity(current_map)!r}"
        )

    for symbol, value in fixture.event_vars:
        if not set_event_var(symbol.removeprefix("VAR_"), value):
            raise ValueError(f"gate fixture references unknown event variable {symbol!r}")

    if fixture.player_coordinates is not None:
        player_object_id = read_symbol("gPlayerAvatar", offset=5, size=1)[0]
        _write_object_coordinates(player_object_id, fixture.player_coordinates)

    runtime_objects = get_runtime_object_table(_fixture_map_identity(current_map))
    for local_id, coordinates in fixture.object_positions:
        matching = tuple(
            item
            for item in runtime_objects
            if item.active and item.local_id == local_id and item.map_id == _fixture_map_identity(current_map)
        )
        if len(matching) != 1:
            raise ValueError(
                f"gate fixture expected one active local ID {local_id} on map {current_map!r}, found {len(matching)}"
            )
        _write_object_coordinates(matching[0].slot, coordinates)

    state_cache.invalidate_runtime_observations()


class GateEdgeValidationMode(BotMode):
    """Use ordinary navigation to exercise one externally armed edge.

    This mode is intentionally opt-in and one-shot. It is suitable for a
    copied profile or a replay fixture; it does not invent a candidate or
    probe an edge when no explicit recorder binding is configured.
    """

    @staticmethod
    def name() -> str:
        return "Gate Edge Validation"

    @staticmethod
    def is_selectable() -> bool:
        from modules.context import context

        return context.rom is not None and context.rom.is_emerald and bool(
            getattr(context.gate_live_observer, "bindings", ())
        )

    def run(self) -> Generator:
        from modules.context import context

        observer = getattr(context, "gate_live_observer", None)
        bindings = tuple(getattr(observer, "bindings", ()))
        if not bindings:
            diagnostic_print("GATE_LIVE_VALIDATION: no explicit binding configured", trace=True)
            context.set_manual_mode(enable_video_and_slow_down=False)
            return
        binding = bindings[0]
        fixture_applied = False
        while True:
            if get_game_state() is not GameState.OVERWORLD:
                yield
                continue
            overworld = perceive_overworld()
            if not overworld.controllable:
                yield
                continue
            fixture = getattr(observer, "fixture", None)
            if fixture is not None and not fixture_applied:
                try:
                    _apply_gate_fixture(
                        fixture,
                        current_map=overworld.map_id,
                        expected_map=binding.edge.source_map,
                    )
                except (AttributeError, KeyError, RuntimeError, TypeError, ValueError) as exc:
                    diagnostic_print(
                        lambda: f"GATE_LIVE_FIXTURE: failed to apply fixture: {exc}",
                        trace=True,
                        prefix="GATE_LIVE_FIXTURE",
                    )
                    add_diagnostic = getattr(observer, "add_diagnostic", None)
                    if callable(add_diagnostic):
                        add_diagnostic(f"fixture application failed: {exc}")
                    flush = getattr(observer, "flush", None)
                    if callable(flush):
                        flush()
                    context.set_manual_mode(enable_video_and_slow_down=False)
                    return
                fixture_applied = True
                diagnostic_print(
                    lambda: (
                        "GATE_LIVE_FIXTURE: applied "
                        f"source_map={fixture.source_map!r} event_vars={fixture.event_vars!r} "
                        f"player_coordinates={fixture.player_coordinates!r} "
                        f"object_positions={fixture.object_positions!r}"
                    ),
                    trace=True,
                    prefix="GATE_LIVE_FIXTURE",
                )
                yield
                continue
            transition = next(
                (
                    item
                    for item in overworld.transitions
                    if item.entry[0] == binding.edge.source_map
                    and item.destination is not None
                    and item.destination[0] == binding.edge.destination_map
                    and item.kind == binding.edge.kind
                ),
                None,
            )
            if transition is None:
                diagnostic_print(
                    lambda: (
                        "GATE_LIVE_VALIDATION: no observed transition "
                        f"candidate={binding.candidate_id!r} edge={binding.edge!r} "
                        f"map={overworld.map_id!r}"
                    ),
                    trace=True,
                )
                context.set_manual_mode(enable_video_and_slow_down=False)
                return
            navigation_goal = NavigationGoal(
                ReachWarp(
                    destination_map=transition.destination[0],
                    destination=transition.destination,
                    warp=transition,
                ),
                constraints=GoalConstraints(),
            )
            diagnostic_print(
                lambda: (
                    "GATE_LIVE_VALIDATION_START: "
                    f"candidate={binding.candidate_id!r} edge={binding.edge!r} "
                    f"transition={transition!r}"
                ),
                trace=True,
            )
            observed_navigation = NavigationAction(
                NavigationActionType.WARP,
                transition.required_facing,
                transition.entry,
                transition.destination,
                transition_kind=transition.kind,
            )
            dynamic_objects = tuple(getattr(overworld, "objects", ()))
            loop = AgentControlLoop(
                lambda: observe_agent(goal=navigation_goal),
                goal=navigation_goal,
                use_movement_batch=False,
            ).run()
            issued = False
            record_issued = getattr(observer, "record_navigation_issued", None)
            if callable(record_issued):
                get_frame_count = getattr(getattr(context, "emulator", None), "get_frame_count", None)
                try:
                    action_frame = get_frame_count() if callable(get_frame_count) else None
                except (AttributeError, RuntimeError, TypeError, ValueError):
                    action_frame = None
                issued = bool(
                    record_issued(
                        observed_navigation,
                        action_frame=action_frame,
                        source_kind="transition",
                        dynamic_objects=dynamic_objects,
                    )
                )
            while True:
                try:
                    next(loop)
                except StopIteration:
                    break
                except NavigationError:
                    record_terminal = getattr(observer, "record_navigation_terminal_outcome", None)
                    if callable(record_terminal):
                        get_frame_count = getattr(getattr(context, "emulator", None), "get_frame_count", None)
                        try:
                            observation_frame = get_frame_count() if callable(get_frame_count) else None
                        except (AttributeError, RuntimeError, TypeError, ValueError):
                            observation_frame = None
                        record_terminal(
                            observed_navigation,
                            "blocked",
                            action_frame=action_frame,
                            observation_frame=observation_frame,
                            source_kind="transition",
                            dynamic_objects=dynamic_objects,
                        )
                    break
                boundary = perceive_overworld()
                if boundary.map_id == transition.destination[0]:
                    # The destination-map observation is itself a terminal
                    # open boundary for this one-shot probe. A copied profile
                    # can cross between two controller observations, so the
                    # fallback records the action frame as unavailable rather
                    # than manufacturing a frame number.
                    record_outcome = getattr(observer, "record_navigation_outcome", None)
                    if callable(record_outcome):
                        get_frame_count = getattr(getattr(context, "emulator", None), "get_frame_count", None)
                        try:
                            observation_frame = get_frame_count() if callable(get_frame_count) else None
                        except (AttributeError, RuntimeError, TypeError, ValueError):
                            observation_frame = None
                        record_outcome(
                            observed_navigation,
                            "open",
                            source_kind="transition",
                            observation_frame=observation_frame,
                        )
                    break
                yield
            diagnostic_print(
                lambda: (
                    "GATE_LIVE_VALIDATION_COMPLETE: "
                    f"candidate={binding.candidate_id!r} observations="
                    f"{len(getattr(observer, 'observations', ()))!r}"
                ),
                trace=True,
            )
            flush = getattr(observer, "flush", None)
            if callable(flush):
                flush()
            context.set_manual_mode(enable_video_and_slow_down=False)
            return


__all__ = ["GateEdgeValidationMode"]
