import unittest
from types import SimpleNamespace
from unittest.mock import patch

from modules.interaction_state import (
    InteractionObservation,
    InteractionPhase,
    InteractionType,
    _observe_field_message_waiting,
    classify_interaction,
    observe_interaction,
)
from modules.memory import GameState


class InteractionStateTests(unittest.TestCase):
    def _observe(self, *, native, script_active=True, lifecycle=True, waiting=False, confirmation=None):
        import modules.interaction_state as interaction_state

        interaction_state._field_message_advance_ready = False
        script = SimpleNamespace(
            is_active=script_active,
            native_function_name=native,
            script_function_name="SomeScript",
        )
        tasks = [SimpleNamespace(symbol="Task_HandleYesNoInput")] if confirmation is not None else []
        observed_confirmation = None
        if confirmation is not None:
            observed_confirmation = SimpleNamespace(
                active=True,
                input_ready=confirmation.input_ready,
                selected=None,
                options=("YES", "NO"),
            )
        with patch.object(interaction_state, "_field_message_lifecycle_active", lifecycle), patch(
            "modules.interaction_state.get_game_state", return_value=GameState.OVERWORLD
        ), patch("modules.interaction_state.get_global_script_context", return_value=script), patch(
            "modules.interaction_state.is_field_message_waiting_for_input", return_value=waiting
        ), patch(
            "modules.interaction_state.task_is_active", return_value=False
        ), patch(
            "modules.interaction_state.is_field_message_task_waiting_for_input", return_value=False
        ), patch(
            "modules.interaction_state.get_tasks", return_value=tasks
        ), patch(
            "modules.interaction_state.player_avatar_is_controllable", return_value=True
        ), patch(
            "modules.nuzlocke.emerald_confirmation.observe_emerald_confirmation",
            return_value=observed_confirmation,
        ):
            return observe_interaction()

    def test_render_wait_is_not_actionable(self):
        observation = self._observe(native="IsFieldMessageBoxHidden")
        self.assertEqual(observation.interaction_phase, InteractionPhase.FIELD_MESSAGE_RENDER_WAIT)
        self.assertFalse(observation.dialogue_waiting)

    def test_emerald_render_rescue_is_limited_to_nurse_rendering_boundary(self):
        import modules.interaction_state as interaction_state

        runtime = {"printer_active": 1, "printer_state": 0}
        script = SimpleNamespace(
            is_active=True,
            native_function_name="IsFieldMessageBoxHidden",
            script_function_name="EventScript_PkmnCenterNurse_ReturnPkmn",
            stack=("Common_EventScript_PkmnCenterNurse",),
        )
        task = SimpleNamespace(data_value=lambda _: 2)

        def read_runtime_symbol(symbol, *args, **kwargs):
            if symbol == "sTextPrinters" and kwargs.get("offset") == 0x1B:
                return bytes((runtime["printer_active"], runtime["printer_state"]))
            if symbol == "gDisableTextPrinters":
                return bytes((0,))
            return bytes(0x24)

        fake_context = SimpleNamespace(
            rom=SimpleNamespace(is_emerald=True, is_rs=False),
            frame=0,
            stutter_trace=None,
        )
        with (
            patch.object(interaction_state, "_field_message_lifecycle_active", True),
            patch.object(interaction_state, "_field_message_advance_ready", False),
            patch.object(interaction_state, "_field_message_render_rescue_pulses", 0),
            patch("modules.interaction_state.context", fake_context),
            patch("modules.interaction_state.get_game_state", return_value=GameState.OVERWORLD),
            patch("modules.interaction_state.get_global_script_context", return_value=script),
            patch("modules.interaction_state.is_field_message_waiting_for_input", return_value=False),
            patch("modules.interaction_state.task_is_active", return_value=True),
            patch("modules.interaction_state.is_field_message_task_waiting_for_input", return_value=False),
            patch("modules.interaction_state.get_task", return_value=task),
            patch("modules.interaction_state.get_tasks", return_value=[]),
            patch("modules.interaction_state.player_avatar_is_controllable", return_value=True),
            patch("modules.interaction_state.read_symbol", side_effect=read_runtime_symbol),
            patch("modules.nuzlocke.emerald_confirmation.observe_emerald_confirmation", return_value=None),
        ):
            self.assertTrue(observe_interaction().field_message_render_rescue_available)

            # Only the exact HandleCharacter boundary is eligible.  Other
            # printer states, including Clear/ScrollStart, must not create a
            # fresh-B action merely because the draw task is still active.
            for printer_state in (1, 2, 3):
                runtime["printer_state"] = printer_state
                self.assertFalse(observe_interaction().field_message_render_rescue_available)

            # A matching printer state is still insufficient once the shared
            # interaction observer has reached the ordinary input boundary.
            script.native_function_name = "WaitForAorBPress"
            with patch("modules.interaction_state.is_field_message_waiting_for_input", return_value=True):
                self.assertFalse(observe_interaction().field_message_render_rescue_available)
            script.native_function_name = "IsFieldMessageBoxHidden"

            # The initial nurse prompt also uses the shared wrapper. Its
            # stack marker makes it safe to accelerate while the menu is not
            # yet installed; the held B is released before choosing YES.
            script.script_function_name = "Std_MsgboxYesNo"
            runtime["printer_state"] = 0
            self.assertTrue(observe_interaction().field_message_render_rescue_available)
            script.script_function_name = "OldaleTown_PokemonCenter_1F_EventScript_Nurse"
            self.assertTrue(observe_interaction().field_message_render_rescue_available)
            script.script_function_name = "EventScript_PkmnCenterNurse_ReturnPkmn"
            interaction_state._field_message_render_rescue_pulses = 0
            observations = [observe_interaction() for _ in range(12)]

        self.assertTrue(all(item.field_message_render_rescue_available for item in observations[:-1]))
        self.assertTrue(observations[-1].field_message_render_rescue_available)
        self.assertEqual(
            observations[-1].metadata["field_message_render_rescue_pulses"],
            12,
        )

        script.script_function_name = "EventScript_PkmnCenterNurse_HealPkmn"
        fake_context.rom.is_emerald = True
        with (
            patch.object(interaction_state, "_field_message_render_rescue_pulses", 0),
            patch("modules.interaction_state.context", fake_context),
            patch("modules.interaction_state.get_game_state", return_value=GameState.OVERWORLD),
            patch("modules.interaction_state.get_global_script_context", return_value=script),
            patch("modules.interaction_state.is_field_message_waiting_for_input", return_value=False),
            patch("modules.interaction_state.task_is_active", return_value=True),
            patch("modules.interaction_state.is_field_message_task_waiting_for_input", return_value=False),
            patch("modules.interaction_state.get_task", return_value=task),
            patch("modules.interaction_state.get_tasks", return_value=[]),
            patch("modules.interaction_state.player_avatar_is_controllable", return_value=True),
            patch("modules.interaction_state.read_symbol", side_effect=read_runtime_symbol),
            patch("modules.nuzlocke.emerald_confirmation.observe_emerald_confirmation", return_value=None),
        ):
            self.assertTrue(observe_interaction().field_message_render_rescue_available)

        # The workaround is Emerald-specific and must not leak to another
        # game even when the rest of the observed state looks identical.
        fake_context.rom.is_emerald = False
        with (
            patch("modules.interaction_state.context", fake_context),
            patch("modules.interaction_state.get_game_state", return_value=GameState.OVERWORLD),
            patch("modules.interaction_state.get_global_script_context", return_value=script),
            patch("modules.interaction_state.is_field_message_waiting_for_input", return_value=False),
            patch("modules.interaction_state.task_is_active", return_value=True),
            patch("modules.interaction_state.is_field_message_task_waiting_for_input", return_value=False),
            patch("modules.interaction_state.get_task", return_value=task),
            patch("modules.interaction_state.get_tasks", return_value=[]),
            patch("modules.interaction_state.player_avatar_is_controllable", return_value=True),
            patch("modules.interaction_state.read_symbol", side_effect=read_runtime_symbol),
            patch("modules.nuzlocke.emerald_confirmation.observe_emerald_confirmation", return_value=None),
        ):
            self.assertFalse(observe_interaction().field_message_render_rescue_available)

    def test_wait_for_a_or_b_is_field_message_input(self):
        observation = self._observe(native="WaitForAorBPress", waiting=True)
        self.assertEqual(observation.interaction_phase, InteractionPhase.FIELD_MESSAGE_INPUT_WAIT)
        self.assertTrue(observation.dialogue_waiting)

    def test_wait_for_a_or_b_without_lifecycle_is_script_wait(self):
        observation = self._observe(native="WaitForAorBPress", lifecycle=False, waiting=False)
        self.assertEqual(observation.interaction_phase, InteractionPhase.SCRIPT_NATIVE_WAIT)
        self.assertFalse(observation.dialogue_waiting)

    def test_yes_no_task_is_choice_phase(self):
        confirmation = SimpleNamespace(active=True, input_ready=True)
        observation = self._observe(native="IsFieldMessageBoxHidden", confirmation=confirmation)
        self.assertEqual(observation.interaction_phase, InteractionPhase.CHOICE_MENU_INPUT_WAIT)
        self.assertTrue(observation.choice_menu_active)
        self.assertTrue(observation.choice_menu_input_ready)
        self.assertEqual(classify_interaction(observation), InteractionType.CHOICE)

    def test_yes_no_script_initial_message_remains_dialogue_until_menu_is_ready(self):
        """A nurse prompt needs A before Emerald creates Task_HandleYesNoInput."""
        import modules.interaction_state as interaction_state

        script = SimpleNamespace(
            is_active=True,
            native_function_name="IsFieldMessageBoxHidden",
            script_function_name="Std_MsgboxYesNo",
        )
        # This reproduces the field-message boundary observed for the nurse:
        # its initial text is advanceable, but the Yes/No task is absent.
        with patch.object(interaction_state, "_field_message_lifecycle_active", True), patch.object(
            interaction_state, "_field_message_advance_ready", True
        ), patch(
            "modules.interaction_state._observe_field_message_waiting", return_value=False
        ), patch("modules.interaction_state.get_game_state", return_value=GameState.OVERWORLD), patch(
            "modules.interaction_state.get_global_script_context", return_value=script
        ), patch("modules.interaction_state.is_field_message_task_waiting_for_input", return_value=False), patch(
            "modules.interaction_state.get_task", return_value=None
        ), patch("modules.interaction_state.player_avatar_is_controllable", return_value=True), patch(
            "modules.nuzlocke.emerald_confirmation.observe_emerald_confirmation", return_value=None
        ):
            observation = observe_interaction()

        self.assertEqual(observation.interaction_phase, InteractionPhase.FIELD_MESSAGE_INPUT_WAIT)
        self.assertFalse(observation.choice_menu_active)
        self.assertEqual(classify_interaction(observation), InteractionType.DIALOGUE)

    def test_yes_no_script_printer_wait_remains_dialogue(self):
        """Emerald keeps Std_MsgboxYesNo at IsFieldMessageBoxHidden for its A wait."""
        import modules.interaction_state as interaction_state

        script = SimpleNamespace(
            is_active=True,
            native_function_name="IsFieldMessageBoxHidden",
            script_function_name="Std_MsgboxYesNo",
        )
        with patch.object(interaction_state, "_field_message_lifecycle_active", True), patch(
            "modules.interaction_state.get_game_state", return_value=GameState.OVERWORLD
        ), patch("modules.interaction_state.get_global_script_context", return_value=script), patch(
            "modules.interaction_state.is_field_message_waiting_for_input", return_value=True
        ), patch("modules.interaction_state.is_field_message_task_waiting_for_input", return_value=False), patch(
            "modules.interaction_state.task_is_active", return_value=True
        ), patch("modules.interaction_state.get_task", return_value=SimpleNamespace(data_value=lambda _: 2)), patch(
            "modules.interaction_state.read_symbol", return_value=bytes((1, 2))
        ), patch("modules.interaction_state.player_avatar_is_controllable", return_value=True), patch(
            "modules.nuzlocke.emerald_confirmation.observe_emerald_confirmation", return_value=None
        ):
            observation = observe_interaction()

        self.assertEqual(observation.interaction_phase, InteractionPhase.FIELD_MESSAGE_INPUT_WAIT)
        self.assertEqual(classify_interaction(observation), InteractionType.DIALOGUE)

    def test_other_script_native_wait_is_not_dialogue(self):
        observation = self._observe(native="WaitForMovementFinish", lifecycle=False, waiting=False)
        self.assertEqual(observation.interaction_phase, InteractionPhase.SCRIPT_NATIVE_WAIT)
        self.assertEqual(classify_interaction(observation), InteractionType.UNKNOWN)

    def test_lifecycle_survives_render_to_input_transition(self):
        import modules.interaction_state as interaction_state

        script = SimpleNamespace(
            is_active=True, native_function_name="IsFieldMessageBoxHidden", script_function_name="Std_MsgboxDefault"
        )
        with patch.object(interaction_state, "_field_message_lifecycle_active", True), patch(
            "modules.interaction_state.get_global_script_context", return_value=script
        ), patch("modules.interaction_state.is_field_message_waiting_for_input", return_value=False), patch(
            "modules.interaction_state.task_is_active", return_value=False
        ):
            self.assertFalse(_observe_field_message_waiting(GameState.OVERWORLD))

        script.native_function_name = "WaitForAorBPress"
        with patch.object(interaction_state, "_field_message_lifecycle_active", True), patch(
            "modules.interaction_state.get_global_script_context", return_value=script
        ), patch("modules.interaction_state.is_field_message_waiting_for_input", return_value=True), patch(
            "modules.interaction_state.task_is_active", return_value=False
        ):
            self.assertTrue(_observe_field_message_waiting(GameState.OVERWORLD))

    def test_script_owned_waitbuttonpress_remains_dialogue_after_message(self):
        import modules.interaction_state as interaction_state

        script = SimpleNamespace(
            is_active=True,
            native_function_name="WaitForAorBPress",
            script_function_name="OldaleTown_PokemonCenter_1F_EventScript_Nurse",
        )
        with patch.object(interaction_state, "_field_message_lifecycle_active", True), patch(
            "modules.interaction_state.get_global_script_context", return_value=script
        ), patch("modules.interaction_state.is_waiting_for_input", return_value=True), patch(
            "modules.interaction_state.is_field_message_waiting_for_input", return_value=True
        ), patch(
            "modules.interaction_state.task_is_active", return_value=False
        ):
            self.assertTrue(_observe_field_message_waiting(GameState.OVERWORLD))

    def test_unknown_coarse_state_still_recognizes_field_dialogue_wait(self):
        script = SimpleNamespace(
            is_active=True,
            native_function_name="WaitForAorBPress",
            script_function_name="Std_MsgboxDefault",
        )
        with patch("modules.interaction_state.get_global_script_context", return_value=script), patch(
            "modules.interaction_state.is_waiting_for_input", return_value=True
        ):
            self.assertTrue(_observe_field_message_waiting(GameState.UNKNOWN))

    def test_unknown_state_without_active_message_is_not_sticky(self):
        script = SimpleNamespace(
            is_active=False,
            native_function_name=None,
            script_function_name=None,
        )
        with patch("modules.interaction_state.get_global_script_context", return_value=script), patch(
            "modules.interaction_state.is_waiting_for_input", return_value=False
        ):
            self.assertFalse(_observe_field_message_waiting(GameState.UNKNOWN))


if __name__ == "__main__":
    unittest.main()
