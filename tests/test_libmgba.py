import queue
import sys
import types
import unittest


class _FakeCore:
    def __init__(self):
        self.inputs = 0
        self.frames = []

    def set_keys(self, _core, inputs):
        self.inputs = inputs
        self.frames.append(inputs)

    def get_keys(self, _core):
        return self.inputs

    def run_frame(self):
        pass


class _FakeTracker:
    time_spent_emulating = 0
    time_spent_total = 0

    def track_frame(self):
        pass


class TestLibmgbaInputTransitions(unittest.TestCase):
    def _emulator(self):
        # The input state machine is independent of the optional native mGBA
        # shared library, which is not present in every test environment.
        # Stub only the native modules so the real LibmgbaEmulator methods are
        # exercised below.
        mgba = types.ModuleType("mgba")
        mgba.ffi = types.SimpleNamespace()
        mgba.lib = types.SimpleNamespace()
        mgba.libmgba_version_string = lambda: "test"
        for name in ("audio", "core", "gba", "image", "log", "png", "vfs"):
            submodule = types.ModuleType(f"mgba.{name}")
            setattr(mgba, name, submodule)
            sys.modules[f"mgba.{name}"] = submodule
        mgba.gba.GBAMemory = object
        sys.modules["mgba"] = mgba

        from modules.libmgba import LibmgbaEmulator

        native_core = _FakeCore()
        emulator = LibmgbaEmulator.__new__(LibmgbaEmulator)
        emulator._core = types.SimpleNamespace(
            _core=types.SimpleNamespace(
                setKeys=lambda _core, inputs: native_core.set_keys(_core, inputs),
                getKeys=lambda _core: native_core.get_keys(_core),
            ),
            run_frame=native_core.run_frame,
        )
        emulator._native_core = native_core
        emulator._performance_tracker = _FakeTracker()
        emulator._pressed_inputs = 0
        emulator._fresh_input_pending = 0
        emulator._fresh_pulse_pending = 0
        emulator._prev_pressed_inputs = 0
        emulator._held_inputs = 0
        emulator._previous_frame_inputs = 0
        emulator._current_frame_inputs = 0
        emulator._gba_audio = types.SimpleNamespace(available=0, clear=lambda: None)
        emulator._last_audio_data = queue.Queue()
        emulator._on_frame_callback = lambda: None
        emulator._throttled = False
        emulator._audio_enabled = False
        return emulator, native_core

    def test_a_press_creates_new_transition_and_neutral_frame_rearms_it(self):
        emulator, native_core = self._emulator()

        emulator.press_button("A")
        emulator.run_single_frame()
        self.assertEqual(native_core.frames[-1], 0x1)  # A_BUTTON
        self.assertEqual(emulator.get_current_frame_inputs(), 0x1)
        self.assertEqual(emulator.get_previous_frame_inputs(), 0)
        self.assertEqual(emulator.get_new_inputs(), 0x1)

        # This is the release frame used by the dialogue handoff workaround.
        emulator.run_single_frame()
        self.assertEqual(native_core.frames[-1], 0)
        self.assertEqual(emulator.get_released_inputs(), 0x1)
        self.assertEqual(emulator.get_new_inputs(), 0)

        # A must be a fresh transition after the neutral frame.
        emulator.press_button("A")
        emulator.run_single_frame()
        self.assertEqual(native_core.frames[-1], 0x1)
        self.assertEqual(emulator.get_new_inputs(), 0x1)

    def test_release_button_removes_held_a_on_next_frame(self):
        emulator, native_core = self._emulator()

        emulator.hold_button("A")
        emulator.run_single_frame()
        self.assertEqual(native_core.frames[-1], 0x1)
        self.assertEqual(emulator.get_new_inputs(), 0x1)

        emulator.release_button("A")
        emulator.run_single_frame()
        self.assertEqual(native_core.frames[-1], 0)
        self.assertEqual(emulator.get_released_inputs(), 0x1)

    def test_fresh_button_inserts_neutral_frame_when_same_input_was_just_applied(self):
        emulator, native_core = self._emulator()

        emulator.press_button("Down")
        emulator.run_single_frame()
        emulator.press_button_fresh("Down")
        emulator.run_single_frame()
        self.assertEqual(native_core.frames[-1], 0)
        self.assertEqual(emulator.get_released_inputs(), 0x80)

        emulator.run_single_frame()
        self.assertEqual(native_core.frames[-1], 0x80)
        self.assertEqual(emulator.get_new_inputs(), 0x80)
