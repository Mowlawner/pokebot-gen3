E2E Test Suite
===============

## Prerequisites

For a development environment, install the test runner separately from the
application's runtime dependencies:

```bash
python -m pip install -r requirements-dev.txt
```

Running `pokebot.py` still installs and checks the application dependencies
declared in `requirements.py`; it no longer installs pytest as a runtime
dependency.

Ensure that you have the following 3 games dumped and that they match
these SHA1 hashes:

1. Pokémon Emerald, English (`f3ae088181bf583e55daf962a92bb46f4f1d07b7`)
2. Pokémon Ruby, English, Rev. 2 (`5b64eacf892920518db4ec664e62a086dd5f5bc8`)
3. Pokémon FireRed, English, Rev. 1 (`dd5945db9b930750cb39d00c84da8571feebf417`)


## Run

From **the repository root**, pytest is the preferred test runner. The
existing `unittest.TestCase` tests remain supported during the migration.

```bash
# Runs all tests collected by pytest
python -m pytest
```

The default command runs the fast unit tier and reports emulator/e2e tests as
skipped. Enable the ROM-backed tiers explicitly:

```bash
# Runs unit tests plus ROM-backed emulator tests
python -m pytest --run-emulator
```

```bash
# Runs the complete local suite, including long-running e2e tests
python -m pytest --run-e2e
```

```bash
# Runs one test by node ID
python -m pytest tests/test_mode_spin.py::TestSpin::test_it_catches_shinies
```

The tier markers are also available for selection:

```bash
# Runs only the fast unit tier
python -m pytest -m unit
```

```bash
# Runs emulator-backed tests (serially)
python -m pytest -m emulator --run-emulator
```

The original unittest commands remain useful while the migration is in
progress:

```bash
# Runs all tests
python -m unittest discover -s tests
````

```bash
# Runs only tests from a particular file:
python -m unittest tests.test_mode_spin
```

```bash
# Runs a single test
python -m unittest tests.test_mode_spin.TestModeSpin.test_it_catches_shinies
```

Tests using the global emulator context, ROM-backed save states, or frame
stepping should not yet be run in parallel. Marker assignment and fixture
isolation are being introduced incrementally.


## Writing tests

Check out one of the existing test files for inspiration, but here are
some things to be aware of:

1. A file with test cases must match the naming pattern `test_*.py`.
2. Inside that file, create a class that inherits from `BotTestCase`.
3. This class can have an arbitrary amount of test methods, but each
   of them must have a name that starts with `test_`.
4. Make sure that any imports from `utility.py` are done with
   `from tests.utility import ...` instead of `from utility import ...`.
5. Any imports from the `modules` directory need to be done _inside the
   test method_ and not outside. This is to ensure that nothing is loaded
   before the test runner can set up its mocks.
6. For tests that require soft resets, the save game must be included
   in the state file. That is **not** the case for regular save states
   that the bot creates. To create such a save state, run the bot in
   debug mode and then use the `Debug` menu and select `Create state
   with save game`.

### A basic template

```python
from tests.utility import BotTestCase, with_save_state, with_frame_timeout

class TestWhatever(BotTestCase):
    @with_save_state("emerald/some_state.ss1")
    @with_frame_timeout(1000)
    def test_something(self): ...

    @with_save_state(["emerald/some_state.ss1", "ruby/some_state.ss1"])
    @with_frame_timeout(1500)
    def test_something_else(self):
        # This will run twice -- once with each state listed in the
        # @with_save_state decorator.
        ...
```
