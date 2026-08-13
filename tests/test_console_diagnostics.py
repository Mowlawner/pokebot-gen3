import types
import unittest
from unittest.mock import patch


class TestDiagnosticPrint(unittest.TestCase):
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
        message = unittest.mock.Mock(return_value="trace")
        context = types.SimpleNamespace(debug=True, debug_trace=False)
        with patch("modules.context.context", context), patch("modules.console.console", sink):
            diagnostic_print(message, trace=True)
        message.assert_not_called()

        context.debug_trace = True
        with patch("modules.context.context", context), patch("modules.console.console", sink):
            diagnostic_print(message, trace=True)
        message.assert_called_once_with()
        sink.print.assert_called_once_with("trace")
