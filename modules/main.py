import queue
import sys
import time
from collections import deque
from typing import Generator

from modules.console import console, diagnostic_print, profile_print
from modules.context import context
from modules.memory import get_game_state
from modules.modes import (
    BotMode,
    BotModeError,
    FrameInfo,
    get_bot_listeners,
    get_bot_mode_by_name,
)
from modules.plugins import plugin_profile_loaded, load_built_in_plugins
from modules.state_cache import state_cache
from modules.stats import StatsDatabase
from modules.tasks import get_global_script_context, get_tasks
from modules.nuzlocke.runtime import NuzlockeRuntime
from modules.nuzlocke.persistence import JsonEventStore
from modules.profiler import count as profile_count
from modules.profiler import (
    enabled as profiling_enabled,
    format_snapshot,
    now as profile_now,
    timing as profile_timing,
)
from modules.stutter_trace import StutterTrace

# Contains a queue of tasks that should be run the next time a frame completes.
# This is currently used by the HTTP server component (which runs in a separate thread) to trigger things
# such as extracting the current party, which need to be done from the main thread.
# Each entry here will be executed exactly once and then removed from the queue.
work_queue: queue.Queue[callable] = queue.Queue()


# Keeps a list of inputs that have been pressed for each frame so that the HTTP server
# can fetch and accumulate them for its `Inputs` stream event.
inputs_each_frame: deque[int] = deque(maxlen=128)


class ManualBotMode(BotMode):
    @staticmethod
    def name() -> str:
        return "Manual"

    def run(self) -> Generator:
        yield


def main_loop() -> None:
    """
    This function is run after the user has selected a profile and the emulator has been started.
    """
    try:
        if context.rom.game_name.startswith("Unsupported "):
            console.print("\n[red bold]You are running an unsupported game![/]")
            console.print(
                "\n[red]This ROM does not appear to be an exact copy of an original Gen3 game.\nIt's possible that is has been modified, or that it got corrupted while dumping the cartridge.\nWhile this might still work, chances are that some or all bot functions will not.[/]"
            )
            console.print("\n[red bold]Please do not ask for support if there are any problem with this game.[/]\n")

        # Built-in plugins are only loaded if some bot configuration actually requires them.
        # Since profile configuration can override global configuration, they can only be
        # loaded at this point where the profile has been loaded and so the full config is
        # available.
        #
        # Regular (user-provided) plugins need to be loaded in `pokebot.py` as early as possible
        # because they might add bot modes.
        load_built_in_plugins()
        plugin_profile_loaded(context.profile)

        context.stats = StatsDatabase(context.profile)
        nuzlocke_event_store = JsonEventStore(context.profile.path / "nuzlocke_events.json")

        if context.config.http.http_server.enable:
            from modules.web.http import start_http_server

            start_http_server(
                host=context.config.http.http_server.ip,
                port=context.config.http.http_server.port,
            )

        context.bot_listeners = get_bot_listeners(context.rom)
        context.nuzlocke_runtime = NuzlockeRuntime(event_sink=nuzlocke_event_store)
        trace_output = context.profile.path / "stutter_trace.jsonl" if context.debug_stutter_trace else None
        context.stutter_trace = StutterTrace(
            enabled=context.debug_stutter_trace,
            threshold_ms=context.stutter_trace_threshold_ms,
            output=trace_output,
        )
        previous_frame_info: FrameInfo | None = None
        last_controller_boundary: tuple | None = None

        while True:
            loop_start = profile_now()
            profile_count("main_loop_iterations")
            # Process work queue, which can be used to get the main thread to access the emulator
            # at a 'safe' time (i.e. not in the middle of emulating a frame.)
            while not work_queue.empty():
                callback = work_queue.get_nowait()
                callback()
                work_queue.task_done()

            context.frame += 1
            trace = context.stutter_trace
            if trace is not None and trace.enabled:
                trace.begin(context.frame, context.emulator.get_frame_count())
                trace_loop_start = trace.now()

            stage_start = profile_now()
            game_state = get_game_state()
            script_context = get_global_script_context()
            script_stack = script_context.stack if script_context is not None and script_context.is_active else []
            task_list = get_tasks()
            if task_list is not None:
                active_tasks = [task.symbol.lower() for task in task_list]
            else:
                active_tasks = []
            if profiling_enabled():
                profile_timing("main_frame_state_reads", stage_start)
            state_reads_elapsed = profile_now() - stage_start if profiling_enabled() else 0

            frame_setup_start = profile_now()
            frame_info = FrameInfo(
                frame_count=context.emulator.get_frame_count(),
                game_state=game_state,
                active_tasks=active_tasks,
                script_stack=script_stack,
                controller_stack=[controller.__qualname__ for controller in context.controller_stack],
                previous_frame=previous_frame_info,
            )

            controller_boundary = (
                context.bot_mode,
                type(context.bot_mode_instance).__name__ if context.bot_mode_instance is not None else None,
                tuple(frame_info.controller_stack),
            )
            if controller_boundary != last_controller_boundary:
                diagnostic_print(
                    lambda: (
                        "CONTROLLER_BOUNDARY: "
                        f"mode={context.bot_mode!r} "
                        f"mode_instance={type(context.bot_mode_instance).__name__ if context.bot_mode_instance is not None else None!r} "
                        f"stack={frame_info.controller_stack!r} "
                        f"game_state={frame_info.game_state!r} "
                        f"active_tasks={frame_info.active_tasks!r} "
                        f"script_stack={frame_info.script_stack!r}"
                    ),
                    trace=True,
                )
                last_controller_boundary = controller_boundary

            # Reset all bot listeners if the emulator has been reset.
            if previous_frame_info is not None and previous_frame_info.frame_count > frame_info.frame_count:
                state_cache.reset()
                context.bot_listeners = get_bot_listeners(context.rom)
                context.nuzlocke_runtime = NuzlockeRuntime(event_sink=nuzlocke_event_store)
            if profiling_enabled():
                profile_timing("main_frame_setup", frame_setup_start)
            frame_setup_elapsed = profile_now() - frame_setup_start if profiling_enabled() else 0

            # Capture normalized state at the application frame boundary,
            # before the emulator advances. This is passive and independent of
            # bot modes, listeners, GUI rendering, and HTTP consumers.
            stage_start = profile_now()
            trace_nuzlocke_start = trace.now() if trace is not None else 0
            context.nuzlocke_runtime.update()
            if trace is not None and trace.enabled:
                trace.duration("nuzlocke_duration_ms", trace_nuzlocke_start)
            if profiling_enabled():
                profile_timing("main_frame_nuzlocke", stage_start)
            nuzlocke_elapsed = profile_now() - stage_start if profiling_enabled() else 0

            new_starters_mode_created = False
            if context.bot_mode == "Manual":
                if not isinstance(context.bot_mode_instance, ManualBotMode):
                    context.emulator.reset_held_buttons()
                context.bot_mode_instance = ManualBotMode()
            elif len(context.controller_stack) == 0:
                is_new_starters_mode = context.bot_mode == "Starters"
                if is_new_starters_mode:
                    diagnostic_print(
                        "[bold yellow]CONTROLLER STARTERS HANDOFF: " "OpeningMode returned; creating StartersMode[/]",
                        trace=True,
                    )
                context.bot_mode_instance = get_bot_mode_by_name(context.bot_mode)()
                context.controller_stack.append(context.bot_mode_instance.run())
                new_starters_mode_created = is_new_starters_mode
                if is_new_starters_mode:
                    diagnostic_print(
                        "[bold yellow]CONTROLLER STARTERS HANDOFF: "
                        f"StartersMode created instance={context.bot_mode_instance!r}; "
                        "iterator appended[/]",
                        trace=True,
                    )

            controller_elapsed = 0
            try:
                stage_start = profile_now()
                trace_controller_start = trace.now() if trace is not None else 0
                for listener in context.bot_listeners.copy():
                    listener.handle_frame(context.bot_mode_instance, frame_info)
                if context.bot_mode == "Manual":
                    context.controller_stack = []
                if len(context.controller_stack) > 0:
                    profile_count("controller_next_calls")
                    if new_starters_mode_created:
                        diagnostic_print(
                            "[bold yellow]CONTROLLER STARTERS HANDOFF: " "advancing newly-created StartersMode[/]",
                            trace=True,
                        )
                    active_controller_qualname = context.controller_stack[-1].__qualname__
                    is_starter_flow_controller = active_controller_qualname in (
                        "StartersMode.run",
                        "BattleListener.fight",
                        "isolate_inputs.<locals>.wrapper_function",
                        "DebugUtil.track.<locals>.wrapper_function",
                    )
                    if is_starter_flow_controller:
                        diagnostic_print(
                            lambda: (
                                "STARTER_FLOW: main_loop before next(controller) "
                                f"controller={active_controller_qualname!r} "
                                f"stack={[controller.__qualname__ for controller in context.controller_stack]!r}"
                            ),
                            trace=True,
                        )
                    try:
                        next(context.controller_stack[-1])
                    finally:
                        if is_starter_flow_controller:
                            diagnostic_print(
                                lambda: (
                                    "STARTER_FLOW: main_loop after next(controller) "
                                    f"controller={active_controller_qualname!r} "
                                    f"stack={[controller.__qualname__ for controller in context.controller_stack]!r}"
                                ),
                                trace=True,
                            )
                    if new_starters_mode_created:
                        diagnostic_print(
                            "[bold yellow]CONTROLLER STARTERS HANDOFF: "
                            "newly-created StartersMode advance returned or raised[/]",
                            trace=True,
                        )
                if profiling_enabled():
                    profile_timing("main_frame_controller", stage_start)
                controller_elapsed = profile_now() - stage_start if profiling_enabled() else 0
                if trace is not None and trace.enabled:
                    trace.duration("controller_duration_ms", trace_controller_start)
            except (StopIteration, GeneratorExit):
                completed_controller = context.controller_stack.pop()
                if completed_controller.__qualname__ in (
                    "StartersMode.run",
                    "BattleListener.fight",
                    "isolate_inputs.<locals>.wrapper_function",
                    "DebugUtil.track.<locals>.wrapper_function",
                ):
                    diagnostic_print(
                        lambda: (
                            "STARTER_FLOW: main_loop popped completed controller "
                            f"controller={completed_controller.__qualname__!r} "
                            f"stack={[controller.__qualname__ for controller in context.controller_stack]!r}"
                        ),
                        trace=True,
                    )
            except BotModeError as e:
                context.emulator.reset_held_buttons()
                context.message = str(e)
                context.set_manual_mode()
            except TimeoutError:
                console.print_exception()
                sys.exit(1)
            except Exception as e:
                console.print_exception()
                context.emulator.reset_held_buttons()
                context.message = f"Internal Bot Error: {str(e)}"
                if context.debug:
                    context.debug_stepping_mode()
                    if hasattr(sys, "gettrace") and sys.gettrace() is not None:
                        breakpoint()
                else:
                    context.set_manual_mode()

            stage_start = profile_now()
            trace_emulator_start = trace.now() if trace is not None else 0
            if trace is not None and trace.enabled:
                trace.mark("emulator_advance_start_wall_ns", time.time_ns())
            inputs_each_frame.append(context.emulator.get_inputs())
            context.emulator.run_single_frame()
            if trace is not None and trace.enabled:
                advance_end_wall_ns = time.time_ns()
                trace.mark("emulator_advance_end_wall_ns", advance_end_wall_ns)
                previous_end_wall_ns = getattr(trace, "_last_advance_end_wall_ns", None)
                advance_start_wall_ns = trace.current.get("emulator_advance_start_wall_ns")
                if previous_end_wall_ns is not None and advance_start_wall_ns is not None:
                    trace.mark(
                        "inter_frame_interval_ms",
                        round((advance_start_wall_ns - previous_end_wall_ns) / 1_000_000, 3),
                    )
                trace._last_advance_end_wall_ns = advance_end_wall_ns
                trace.duration("emulator_advancement_duration_ms", trace_emulator_start)
                trace.duration("total_frame_duration_ms", trace_loop_start)
                trace.mark("application_frame", context.frame)
                trace.mark("emulator_frame", context.emulator.get_frame_count())
                trace.finish()
            if profiling_enabled():
                profile_timing("main_frame_emulator", stage_start)
                profile_timing("main_frame_total", loop_start)
                emulator_elapsed = profile_now() - stage_start
                total_elapsed = profile_now() - loop_start
                if total_elapsed >= 20_000_000:
                    profile_print(
                        lambda: (
                            f"FRAME_STALL frame={context.frame} emulator_frame={context.emulator.get_frame_count()} "
                            f"total_ms={total_elapsed / 1_000_000:.3f} "
                            f"state_reads_ms={state_reads_elapsed / 1_000_000:.3f} "
                            f"setup_ms={frame_setup_elapsed / 1_000_000:.3f} "
                            f"nuzlocke_ms={nuzlocke_elapsed / 1_000_000:.3f} "
                            f"controller_ms={controller_elapsed / 1_000_000:.3f} "
                            f"emulator_ms={emulator_elapsed / 1_000_000:.3f} "
                            f"stack_depth={len(context.controller_stack)} "
                            f"profile={format_snapshot()}"
                        )
                    )
            previous_frame_info = frame_info
            previous_frame_info.previous_frame = None

    except SystemExit:
        raise
    except Exception:
        console.print_exception(show_locals=True)
        sys.exit(1)
