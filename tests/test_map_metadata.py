import unittest
import struct
from types import SimpleNamespace
from unittest.mock import patch

from modules.map import (
    MapMetadata,
    MapLocation,
    ObjectEvent,
    ObjectEventTemplate,
    get_map_data,
    get_live_map_id,
    get_map_metadata,
    inspect_map_connections,
    observe_live_map_identity,
)


class TestMapMetadata(unittest.TestCase):
    def setUp(self):
        self.rom_a = SimpleNamespace(id="rom-a")
        self.rom_b = SimpleNamespace(id="rom-b")
        self.map_a = (1, 2)
        self.map_b = (1, 3)
        self.metadata_a = MapMetadata(
            self.map_a, b"header-a", struct.pack("<II", 30, 20), b"events-a", (), (), (), (), ()
        )
        self.metadata_b = MapMetadata(
            self.map_b, b"header-b", struct.pack("<II", 40, 25), b"events-b", (), (), (), (), ()
        )

    def test_metadata_is_built_once_per_rom_and_map(self):
        with (
            patch("modules.map.context", SimpleNamespace(rom=self.rom_a)),
            patch("modules.map._map_metadata_cache", {}),
            patch("modules.map._map_header_cache", {"rom-a": {self.map_a: b"header-a"}}),
            patch("modules.map._build_map_metadata", return_value=self.metadata_a) as build,
        ):
            first = get_map_metadata(self.map_a)
            second = get_map_metadata(self.map_a)

        self.assertIs(first, second)
        build.assert_called_once_with(self.map_a, b"header-a")

    def test_location_views_share_metadata_but_keep_positions(self):
        first = MapLocation(b"header", *self.map_a, (2, 3), metadata=self.metadata_a)
        second = MapLocation(b"header", *self.map_a, (8, 9), metadata=self.metadata_a)

        self.assertIs(first._metadata, second._metadata)
        self.assertEqual(first.local_position, (2, 3))
        self.assertEqual(second.local_position, (8, 9))
        self.assertEqual(first.map_size, second.map_size)

    def test_get_map_data_returns_new_position_view_with_shared_metadata(self):
        with (
            patch("modules.map.context", SimpleNamespace(rom=self.rom_a)),
            patch("modules.map._map_header_cache", {"rom-a": {self.map_a: b"header-a"}}),
            patch("modules.map.get_map_metadata", return_value=self.metadata_a) as metadata,
        ):
            first = get_map_data(self.map_a, (1, 1))
            second = get_map_data(self.map_a, (4, 5))

        self.assertIs(first._metadata, self.metadata_a)
        self.assertIs(second._metadata, self.metadata_a)
        self.assertEqual(first.local_position, (1, 1))
        self.assertEqual(second.local_position, (4, 5))
        self.assertEqual(metadata.call_count, 2)

    def test_map_change_and_rom_change_use_separate_metadata(self):
        with (
            patch("modules.map.context", SimpleNamespace(rom=self.rom_a)),
            patch("modules.map._map_metadata_cache", {}),
            patch(
                "modules.map._map_header_cache",
                {
                    "rom-a": {self.map_a: b"header-a", self.map_b: b"header-b"},
                    "rom-b": {self.map_a: b"header-other"},
                },
            ),
            patch(
                "modules.map._build_map_metadata", side_effect=[self.metadata_a, self.metadata_b, self.metadata_a]
            ) as build,
        ):
            map_a = get_map_metadata(self.map_a)
            map_b = get_map_metadata(self.map_b)

            # A separate ROM identity must not reuse ROM A's map metadata.
            with patch("modules.map.context", SimpleNamespace(rom=self.rom_b)):
                other_rom_map_a = get_map_metadata(self.map_a)

        self.assertIs(map_a, self.metadata_a)
        self.assertIs(map_b, self.metadata_b)
        self.assertIs(other_rom_map_a, self.metadata_a)
        self.assertEqual(build.call_count, 3)

    def test_live_map_id_populates_its_own_header_index(self):
        header = bytes(range(0x1C))
        header_cache = {}
        with (
            patch("modules.map.context", SimpleNamespace(rom=self.rom_a)),
            patch("modules.map._map_header_cache", header_cache),
            patch("modules.map.get_map_data") as warm,
            patch("modules.map.read_symbol", return_value=header),
        ):
            warm.side_effect = lambda *_: header_cache.update({"rom-a": {self.map_a: header}})
            self.assertEqual(get_live_map_id(), self.map_a)
        warm.assert_called_once_with((0, 0), (0, 0))

    def test_live_map_id_uses_unique_stable_pointer_identity(self):
        stable = bytes(range(0x10))
        cached_header = stable + bytes(0x0C)
        live_header = stable + bytes([0xFF]) * 0x0C
        with (
            patch("modules.map.context", SimpleNamespace(rom=self.rom_a)),
            patch("modules.map._map_header_cache", {"rom-a": {self.map_a: cached_header}}),
            patch("modules.map.read_symbol", return_value=live_header),
        ):
            self.assertEqual(get_live_map_id(), self.map_a)

    def test_live_map_identity_rejects_ambiguous_header_match(self):
        shared_header = bytes(range(0x1C))
        with (
            patch("modules.map.context", SimpleNamespace(rom=self.rom_a)),
            patch(
                "modules.map._map_header_cache",
                {"rom-a": {self.map_a: shared_header, self.map_b: shared_header}},
            ),
            patch("modules.map.read_symbol", return_value=shared_header),
        ):
            identity = observe_live_map_identity()

        self.assertIsNone(identity.map_id)
        self.assertEqual(identity.source, "unresolved")
        self.assertEqual(identity.candidates, (self.map_a, self.map_b))

    def test_object_template_lookup_uses_metadata_without_get_map_data(self):
        data = bytearray(0x24)
        data[8] = 2
        data[9] = self.map_a[1]
        data[10] = self.map_a[0]
        event = ObjectEvent(bytes(data))
        template = SimpleNamespace(local_id=2)
        metadata = SimpleNamespace(object_template=lambda local_id: template if local_id == 2 else None)

        with (
            patch("modules.map.get_map_metadata", return_value=metadata),
            patch("modules.map.get_map_data", side_effect=AssertionError("redundant map lookup")),
        ):
            self.assertIs(event.object_event_template, template)

    def test_object_template_index_preserves_first_duplicate(self):
        first = SimpleNamespace(local_id=2)
        duplicate = SimpleNamespace(local_id=2)
        metadata = MapMetadata(
            self.map_a,
            b"header-a",
            struct.pack("<II", 30, 20),
            b"events-a",
            (),
            (),
            (),
            (),
            (first, duplicate),
        )

        self.assertIs(metadata.object_template(2), first)
        self.assertIs(metadata.object_template(99), None)

    def test_inspect_map_connections_reports_raw_and_decoded_records(self):
        map_id = (0, 16)
        destination = (0, 10)
        header = bytearray(0x1C)
        header[0x0C:0x10] = (0x2000).to_bytes(4, "little")
        record = bytes([2, 0, 0, 0]) + (0).to_bytes(4, "little", signed=True) + bytes([0, 10, 0, 0])
        map_view = SimpleNamespace(map_name="ROUTE 101")
        destination_view = SimpleNamespace(map_name="OLDALE TOWN")
        emulator = SimpleNamespace(
            read_bytes=lambda address, size: {
                0x1000: (0x3000).to_bytes(4, "little"),
                0x2000: (1).to_bytes(4, "little") + (0x4000).to_bytes(4, "little"),
                0x4000: record,
            }[address]
        )
        rom = SimpleNamespace(
            id="rom-a",
            game_name="Pokémon Emerald (E)",
            game_code="BPE",
            revision=0,
            language=SimpleNamespace(name="English"),
        )
        with (
            patch("modules.map.context", SimpleNamespace(rom=rom, emulator=emulator)),
            patch("modules.map._map_header_cache", {"rom-a": {map_id: bytes(header)}}),
            patch("modules.map.get_map_data", side_effect=[map_view, destination_view]),
            patch("modules.map.read_symbol", return_value=(0x0FC0).to_bytes(4, "little")),
        ):
            result = inspect_map_connections(map_id)

        self.assertEqual(result["connections"]["count"], 1)
        self.assertEqual(result["connections"]["list_pointer"], 0x2000)
        self.assertEqual(result["connections"]["records"][0]["raw_direction"], 2)
        self.assertEqual(result["connections"]["records"][0]["direction"], "North")
        self.assertEqual(result["connections"]["records"][0]["destination_name"], "OLDALE TOWN")

    def test_object_template_script_symbol_is_resolved_once(self):
        data = bytearray(24)
        data[16:20] = (0x12345678).to_bytes(4, "little")
        template = ObjectEventTemplate(bytes(data))
        with patch("modules.map.get_symbol_name", return_value="TestScript") as symbol:
            self.assertEqual(template.script_symbol, "TestScript")
            self.assertEqual(template.script_symbol, "TestScript")
        symbol.assert_called_once_with(0x12345678, pretty_name=True)


if __name__ == "__main__":
    unittest.main()
