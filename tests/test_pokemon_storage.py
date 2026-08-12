import importlib
import sys
import types
import unittest
from unittest.mock import Mock


class _CacheItem:
    age_in_frames = 1
    value = None


class TestPokemonStorageReader(unittest.TestCase):
    def _load_reader(self, *, offset):
        emulator = types.SimpleNamespace(read_bytes=Mock(return_value=bytes(33872)))
        fake_modules = {
            "modules.context": types.SimpleNamespace(
                context=types.SimpleNamespace(
                    rom=types.SimpleNamespace(is_rs=True),
                    emulator=emulator,
                )
            ),
            "modules.game": types.SimpleNamespace(
                decode_string=lambda value: "",
                get_symbol=Mock(return_value=(offset, 33872)),
            ),
            "modules.memory": types.SimpleNamespace(
                read_symbol=Mock(),
                unpack_uint32=lambda value: 0,
            ),
            "modules.pokemon": types.SimpleNamespace(Pokemon=object, Species=object),
            "modules.state_cache": types.SimpleNamespace(
                state_cache=types.SimpleNamespace(pokemon_storage=_CacheItem())
            ),
        }
        old_modules = {name: sys.modules.get(name) for name in fake_modules}
        sys.modules.update(fake_modules)
        sys.modules.pop("modules.pokemon_storage", None)
        module = importlib.import_module("modules.pokemon_storage")
        return module, emulator, old_modules

    @staticmethod
    def _restore(module, old_modules):
        sys.modules.pop("modules.pokemon_storage", None)
        for name, old_module in old_modules.items():
            if old_module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old_module

    def test_zero_storage_address_is_unavailable_without_reading_memory(self):
        module, emulator, old_modules = self._load_reader(offset=0)
        try:
            self.assertIsNone(module.get_pokemon_storage())
            emulator.read_bytes.assert_not_called()
        finally:
            self._restore(module, old_modules)

    def test_nonzero_storage_address_returns_readable_empty_storage(self):
        module, emulator, old_modules = self._load_reader(offset=0x02000000)
        try:
            storage = module.get_pokemon_storage()
            self.assertIsInstance(storage, module.PokemonStorage)
            emulator.read_bytes.assert_called_once_with(0x02000000, 33872)
        finally:
            self._restore(module, old_modules)


if __name__ == "__main__":
    unittest.main()
