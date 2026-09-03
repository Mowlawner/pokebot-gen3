"""Temporary diagnostic wrapper for sampling the emulated ARM CPU during a stall."""

import ctypes
import os
import runpy
import subprocess
import sys
import threading
import time

from mgba.core import Core
from mgba import ffi
from mgba.gba import GBA

_run_frame = Core.run_frame
_gba_init = GBA.__init__


def _gba_init_with_trace(self, native):
    _gba_init(self, native)
    key_read_count = 0

    def io_snapshot():
        """Read the hardware state relevant to Emerald's WaitForVBlank loop."""
        try:
            return {
                "dispstat": self.memory.io.u16[0x04],
                "vcount": self.memory.io.u16[0x06],
                "ime": self.memory.io.u16[0x208],
                "ie": self.memory.io.u16[0x200],
                "if": self.memory.io.u16[0x202],
                "intr_main_flag": self.memory.iwram.u16[0x22DC],
            }
        except Exception as error:  # pragma: no cover - diagnostic-only fallback
            return {"error": repr(error)}

    def keys_read():
        nonlocal key_read_count
        try:
            frame_counter = self._core.frameCounter(self._core)
            if 413973 <= frame_counter <= 413976 and key_read_count < 20:
                key_read_count += 1
                print(
                    "NATIVE_KEYS_READ: "
                    f"frame={frame_counter} pc={self.cpu.pc!r} "
                    f"sp={self.cpu.sp!r} lr={self.cpu.lr!r} "
                    f"cpsr={self.cpu.cpsr!r} "
                    f"cpsr_i={self.cpu.cpsr.i!r} cpsr_packed={self.cpu.cpsr.packed!r} "
                    f"cycles={self.cpu._native.cycles!r} "
                    f"next_event={self.cpu._native.nextEvent!r}",
                    f" io={io_snapshot()!r}",
                    flush=True,
                )
        except Exception as error:  # pragma: no cover - diagnostic-only fallback
            print(f"NATIVE_KEYS_READ: register read failed: {error!r}", flush=True)

    self._callbacks.keys_read.append(keys_read)

    def video_frame_started():
        try:
            frame_counter = self._core.frameCounter(self._core)
            if 413973 <= frame_counter <= 413976:
                print(
                    "NATIVE_VIDEO_FRAME_STARTED: "
                    f"pc={self.cpu.pc!r} sp={self.cpu.sp!r} lr={self.cpu.lr!r} "
                    f"cpsr={self.cpu.cpsr!r} "
                    f"cpsr_i={self.cpu.cpsr.i!r} cpsr_packed={self.cpu.cpsr.packed!r} "
                    f"cycles={self.cpu._native.cycles!r} next_event={self.cpu._native.nextEvent!r} "
                    f"io={io_snapshot()!r}",
                    flush=True,
                )
        except Exception as error:  # pragma: no cover - diagnostic-only fallback
            print(f"NATIVE_VIDEO_FRAME_STARTED: register read failed: {error!r}", flush=True)

    def video_frame_ended():
        try:
            frame_counter = self._core.frameCounter(self._core)
            if 413973 <= frame_counter <= 413976:
                print(
                    "NATIVE_VIDEO_FRAME_ENDED: "
                    f"pc={self.cpu.pc!r} sp={self.cpu.sp!r} lr={self.cpu.lr!r} "
                    f"cpsr={self.cpu.cpsr!r} "
                    f"cpsr_i={self.cpu.cpsr.i!r} cpsr_packed={self.cpu.cpsr.packed!r} "
                    f"cycles={self.cpu._native.cycles!r} next_event={self.cpu._native.nextEvent!r} "
                    f"io={io_snapshot()!r}",
                    flush=True,
                )
        except Exception as error:  # pragma: no cover - diagnostic-only fallback
            print(f"NATIVE_VIDEO_FRAME_ENDED: register read failed: {error!r}", flush=True)

    self._callbacks.video_frame_started.append(video_frame_started)
    self._callbacks.video_frame_ended.append(video_frame_ended)


GBA.__init__ = _gba_init_with_trace


def _run_frame_with_watchdog(self):
    stop = threading.Event()
    started = time.monotonic()
    native_core = self._core
    frame_counter = native_core.frameCounter(native_core)
    watch_frame = 413973 <= frame_counter <= 413976
    if watch_frame:
        native_cpu = getattr(self, "cpu", None)._native
        print(
            "NATIVE_RUN_BEGIN: "
            f"frame={frame_counter} pc={getattr(self.cpu, 'pc', None)!r} "
            f"keys_active={self._native.keysActive!r} keys_last={self._native.keysLast!r} "
            f"cycles={native_cpu.cycles!r} next_event={native_cpu.nextEvent!r}",
            flush=True,
        )

    def sample_cpu():
        if watch_frame:
            print("NATIVE_WATCHDOG_THREAD_STARTED", flush=True)
        if stop.wait(0.75):
            return
        cpu = getattr(self, "cpu", None)
        while not stop.wait(0.5):
            try:
                native_cpu = getattr(cpu, "_native", None)
                gprs = getattr(cpu, "gprs", None)
                registers = tuple(gprs[index] for index in range(15)) if gprs is not None else ()
                print(
                    "NATIVE_RUN_WATCHDOG: "
                    f"elapsed={time.monotonic() - started:.3f} "
                    f"pc={getattr(cpu, 'pc', None)!r} "
                    f"sp={getattr(cpu, 'sp', None)!r} "
                    f"lr={getattr(cpu, 'lr', None)!r} "
                    f"cpsr={getattr(cpu, 'cpsr', None)!r} "
                    f"cpsr_i={getattr(getattr(cpu, 'cpsr', None), 'i', None)!r} "
                    f"cpsr_packed={getattr(getattr(cpu, 'cpsr', None), 'packed', None)!r} "
                    f"cycles={getattr(native_cpu, 'cycles', None)!r} "
                    f"next_event={getattr(native_cpu, 'nextEvent', None)!r} "
                    f"halted={getattr(native_cpu, 'halted', None)!r} "
                    f"gprs={registers!r}",
                    flush=True,
                )
            except Exception as error:  # pragma: no cover - diagnostic-only fallback
                print(f"NATIVE_RUN_WATCHDOG: register read failed: {error!r}", flush=True)

    watcher = threading.Thread(target=sample_cpu, name="native-run-watchdog", daemon=True)
    watcher.start()
    sampler = None
    if frame_counter == 413976:
        sample_path = "/tmp/pokebot-mart-native-direct-sample.txt"
        sampler = subprocess.Popen(
            [
                "/bin/sh",
                "-c",
                f"/bin/sleep 2\n/usr/bin/sample {os.getpid()} 5 -file {sample_path}",
            ]
        )
    try:
        # CFFI's generated call keeps the GIL, preventing the Python watchdog
        # from running. Calling the same mCore vtable entry through ctypes
        # releases it for the duration of the native call.
        run_frame = ctypes.CFUNCTYPE(None, ctypes.c_void_p)(int(ffi.cast("uintptr_t", native_core.runFrame)))
        core_pointer = ctypes.c_void_p(int(ffi.cast("uintptr_t", native_core)))
        return run_frame(core_pointer)
    finally:
        stop.set()
        if sampler is not None:
            sampler.wait(timeout=15)


Core.run_frame = _run_frame_with_watchdog
sys.argv = ["pokebot.py", *sys.argv[1:]]
runpy.run_path("pokebot.py", run_name="__main__")
