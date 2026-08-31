import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch


class PokemonPartyTests(unittest.TestCase):
    def test_nurse_render_invalid_slot_uses_cached_party_without_peeking(self):
        from modules.pokemon_party import Party, get_party
        from modules.state_cache import state_cache

        frame = {"value": 10}
        emulator = SimpleNamespace(
            get_frame_count=lambda: frame["value"],
            _core=SimpleNamespace(run_frame=Mock()),
        )
        fake_context = SimpleNamespace(
            rom=SimpleNamespace(is_emerald=True),
            emulator=emulator,
        )
        script = SimpleNamespace(
            is_active=True,
            script_function_name="EventScript_PkmnCenterNurse_ReturnPkmn",
            native_function_name="IsFieldMessageBoxHidden",
        )
        task = SimpleNamespace(data_value=lambda index: 2)
        cached_party = Party([])

        state_cache.reset()
        self.addCleanup(state_cache.reset)
        with (
            patch("modules.pokemon_party.context", fake_context),
            patch("modules.state_cache.context", fake_context),
            patch("modules.pokemon_party.get_global_script_context", return_value=script),
            patch("modules.pokemon_party.get_task", return_value=task),
            patch(
                "modules.pokemon_party.read_symbol",
                side_effect=lambda symbol, **kwargs: b"\x01" if symbol == "gPlayerPartyCount" else b"\x01" + bytes(99),
            ),
        ):
            state_cache.party = cached_party
            frame["value"] += 1

            self.assertIs(get_party(), cached_party)

        emulator._core.run_frame.assert_not_called()
