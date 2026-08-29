"""Pytest boundaries for the mixed unit/emulator test suite.

The repository is still compatible with unittest.  This file adds pytest
collection policy without requiring existing test classes to be rewritten.
Tests using the shared BotTestCase harness are conservatively classified as
emulator tests because they may mutate global emulator state or require ROM
fixtures.
"""

from __future__ import annotations

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("nuzbot test tiers")
    group.addoption(
        "--run-emulator",
        action="store_true",
        default=False,
        help="run ROM-backed emulator tests (disabled by default)",
    )
    group.addoption(
        "--run-e2e",
        action="store_true",
        default=False,
        help="run long-running end-to-end tests (also enables emulator tests)",
    )


def _inherits_from_bot_test_case(item: pytest.Item) -> bool:
    test_class = getattr(item, "cls", None)
    return test_class is not None and any(base.__name__ == "BotTestCase" for base in test_class.__mro__)


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    run_emulator = config.getoption("--run-emulator") or config.getoption("--run-e2e")
    run_e2e = config.getoption("--run-e2e")
    skip_emulator = pytest.mark.skip(reason="emulator tier disabled; pass --run-emulator")
    skip_e2e = pytest.mark.skip(reason="e2e tier disabled; pass --run-e2e")

    for item in items:
        # libmgba is imported inside its test methods, so it cannot be
        # identified through BotTestCase. Keep it in the emulator tier
        # explicitly because it needs the native mGBA artifact.
        emulator_module = item.fspath.basename == "test_libmgba.py"
        if _inherits_from_bot_test_case(item) or emulator_module:
            item.add_marker("emulator")
        if item.get_closest_marker("emulator") and not run_emulator:
            item.add_marker(skip_emulator)
        if item.get_closest_marker("e2e") and not run_e2e:
            item.add_marker(skip_e2e)
        if not item.get_closest_marker("emulator") and not item.get_closest_marker("e2e"):
            item.add_marker("unit")
