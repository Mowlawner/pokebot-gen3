import types
import unittest
from unittest.mock import patch


class TestDiagnosticPrint(unittest.TestCase):
    def test_campaign_frontier_is_normal_rich_output(self):
        from modules.console import print_campaign_frontier

        sink = types.SimpleNamespace(print=unittest.mock.Mock())
        with patch("modules.console.console", sink):
            print_campaign_frontier(
                ("receive_pokedex",),
                ("obtain_encounter:0:17",),
                selected="obtain_encounter:0:17",
                status="ready",
                reason="nearest eligible encounter",
            )

        sink.print.assert_called_once()
        rendered = sink.print.call_args.args[0]
        self.assertIn("Campaign Objective Frontier", rendered.title)

    def test_profile_output_is_disabled_without_profile_flag(self):
        from modules.console import profile_print

        sink = types.SimpleNamespace(print=unittest.mock.Mock())
        message = unittest.mock.Mock(return_value="unused")
        context = types.SimpleNamespace(debug=False, debug_trace=True, debug_profile=False)
        with patch("modules.context.context", context), patch("modules.console.console", sink):
            profile_print(message)

        message.assert_not_called()
        sink.print.assert_not_called()

    def test_profile_output_does_not_require_trace_flag(self):
        from modules.console import profile_print

        sink = types.SimpleNamespace(print=unittest.mock.Mock())
        message = unittest.mock.Mock(return_value="profile")
        context = types.SimpleNamespace(debug=False, debug_trace=False, debug_profile=True)
        with patch("modules.context.context", context), patch("modules.console.console", sink):
            profile_print(message)

        message.assert_called_once_with()
        sink.print.assert_called_once_with("profile")

    def test_disabled_diagnostic_does_not_format_message(self):
        from modules.console import diagnostic_print

        sink = types.SimpleNamespace(print=unittest.mock.Mock())
        message = unittest.mock.Mock(return_value="unused")
        context = types.SimpleNamespace(debug=False, debug_trace=False)
        with patch("modules.context.context", context), patch("modules.console.console", sink):
            diagnostic_print(message)

        message.assert_not_called()
        sink.print.assert_not_called()

    def test_trace_requires_explicit_trace_level(self):
        from modules.console import diagnostic_print

        sink = types.SimpleNamespace(print=unittest.mock.Mock())
        message = unittest.mock.Mock(return_value="CAMPAIGN_RECOVERY_SOURCE: trace")
        context = types.SimpleNamespace(debug=True, debug_trace=False)
        with patch("modules.context.context", context), patch("modules.console.console", sink):
            diagnostic_print(message, trace=True, prefix="CAMPAIGN_RECOVERY_SOURCE")
        message.assert_not_called()

        context.debug_trace = True
        with patch("modules.context.context", context), patch("modules.console.console", sink):
            diagnostic_print(message, trace=True, prefix="CAMPAIGN_RECOVERY_SOURCE")
        message.assert_called_once_with()
        sink.print.assert_called_once_with("CAMPAIGN_RECOVERY_SOURCE: trace")

    def test_filtered_lazy_trace_does_not_evaluate_message(self):
        from modules.console import diagnostic_print

        sink = types.SimpleNamespace(print=unittest.mock.Mock())
        message = unittest.mock.Mock(return_value="CAMPAIGN_PLAN_TRACE: expensive")
        context = types.SimpleNamespace(debug=True, debug_trace=True)
        with patch("modules.context.context", context), patch("modules.console.console", sink):
            diagnostic_print(message, trace=True, prefix="UNSELECTED_TRACE")

        message.assert_not_called()
        sink.print.assert_not_called()
