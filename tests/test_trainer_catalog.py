import unittest

from modules.trainer_catalog import parse_trainer_catalog, resolve_trainerbattle_id, resolve_trainerbattle_id_from_script


class TrainerCatalogTests(unittest.TestCase):
    def test_parse_default_party_records_and_names(self):
        table = bytearray(80)
        table[0] = 0
        table[1] = 7
        table[4:16] = b"Calvin\xff\x00\x00\x00\x00\x00"
        table[32:36] = (1).to_bytes(4, "little")
        table[36:40] = (0x08000100).to_bytes(4, "little")
        party = bytearray(8)
        party[2:4] = (5).to_bytes(2, "little")
        party[4:6] = (286).to_bytes(2, "little")

        records = parse_trainer_catalog(
            bytes(table),
            party_reader=lambda address, size: bytes(party[:size]) if address == 0x08000100 else b"",
            name_decoder=lambda value: value.split(b"\xff", 1)[0].decode("ascii"),
        )

        self.assertEqual(len(records), 2)
        self.assertEqual(records[0].name, "Calvin")
        self.assertEqual(records[0].party[0].species_index, 286)
        self.assertEqual(records[0].party[0].level, 5)
        self.assertEqual(records[0].party[0].held_item_index, 0)
        self.assertEqual(records[1].party, ())

    def test_parse_custom_moves_and_rejects_invalid_party(self):
        table = bytearray(40)
        table[0] = 1
        table[32:36] = (1).to_bytes(4, "little")
        table[36:40] = (0x08000200).to_bytes(4, "little")
        party = bytearray(14)
        party[2:4] = (8).to_bytes(2, "little")
        party[4:6] = (286).to_bytes(2, "little")
        for index, move in enumerate((33, 45, 0, 0)):
            party[6 + index * 2 : 8 + index * 2] = move.to_bytes(2, "little")

        records = parse_trainer_catalog(
            bytes(table),
            party_reader=lambda address, size: bytes(party[:size]),
        )

        self.assertEqual(records[0].party[0].moves, (33, 45, 0, 0))
        self.assertIsNone(resolve_trainerbattle_id(b"\x66\x5c\x3e\x01"))
        self.assertEqual(resolve_trainerbattle_id(b"\x5c\x3e\x01", catalog_size=400), 318)
        self.assertIsNone(resolve_trainerbattle_id(b"\x5c\x3e\x01", catalog_size=10))
        self.assertEqual(resolve_trainerbattle_id(b"\x5c\x03\x17\x02", catalog_size=855), 535)

    def test_parse_held_item_default_moves_without_inventing_custom_moves(self):
        table = bytearray(40)
        table[0] = 2  # held item, default moves
        table[32:36] = (1).to_bytes(4, "little")
        table[36:40] = (0x08000200).to_bytes(4, "little")
        party = bytearray(8)
        party[2:4] = (7).to_bytes(2, "little")
        party[4:6] = (288).to_bytes(2, "little")  # Zigzagoon
        party[6:8] = (110).to_bytes(2, "little")  # held item, not a move

        records = parse_trainer_catalog(
            bytes(table),
            party_reader=lambda _address, size: bytes(party[:size]),
        )

        assert records[0] is not None
        assert records[0].party[0].held_item_index == 110
        assert records[0].party[0].moves == ()

    def test_parse_tries_compatible_layout_when_primary_party_reader_rejects_it(self):
        table = bytearray(40)
        table[0] = 1
        table[32:36] = (1).to_bytes(4, "little")
        table[36:40] = (0x08000300).to_bytes(4, "little")
        party = bytearray(14)
        party[2:4] = (8).to_bytes(2, "little")
        party[4:6] = (286).to_bytes(2, "little")
        for index, move in enumerate((33, 45, 0, 0)):
            party[6 + index * 2 : 8 + index * 2] = move.to_bytes(2, "little")

        records = parse_trainer_catalog(
            bytes(table),
            party_reader=lambda _address, size: bytes(party) if size == 14 else b"",
        )

        assert records[0] is not None
        assert records[0].party[0].moves == (33, 45, 0, 0)

    def test_resolve_wrapper_trainerbattle_beyond_short_prefix(self):
        script = b"\x01" * 96 + b"\x5c\x03\x17\x02"
        self.assertEqual(resolve_trainerbattle_id_from_script(script, catalog_size=855), 535)


if __name__ == "__main__":
    unittest.main()
