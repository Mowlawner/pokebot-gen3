import unittest
from types import SimpleNamespace
from unittest.mock import patch


class MartBuyableItemsTests(unittest.TestCase):
    def test_buyable_item_scan_stops_at_item_none(self):
        from modules import mart

        reads = iter((b"\x01\x00", b"\x02\x00", b"\x00\x00"))
        fake_context = SimpleNamespace(
            rom=SimpleNamespace(is_emerald=True, is_frlg=False, is_rs=False),
            emulator=SimpleNamespace(read_bytes=lambda address, length: next(reads)),
        )
        first = SimpleNamespace(name="first")
        second = SimpleNamespace(name="second")
        with (
            patch.object(mart, "context", fake_context),
            patch.object(mart, "read_symbol", return_value=b"\x00\x10\x00\x02"),
            patch.object(mart, "get_item_by_index", side_effect=(first, second)),
        ):
            self.assertEqual(mart.get_mart_buyable_items(), [first, second])

    def test_buyable_item_scan_is_bounded_without_item_none(self):
        from modules import mart

        fake_context = SimpleNamespace(
            rom=SimpleNamespace(is_emerald=True, is_frlg=False, is_rs=False),
            emulator=SimpleNamespace(read_bytes=lambda address, length: b"\x01\x00"),
        )
        with (
            patch.object(mart, "context", fake_context),
            patch.object(mart, "read_symbol", return_value=b"\x00\x10\x00\x02"),
            patch.object(mart, "get_item_by_index", return_value=SimpleNamespace(name="item")),
        ):
            with self.assertRaisesRegex(RuntimeError, "did not contain an ITEM_NONE terminator"):
                mart.get_mart_buyable_items()


if __name__ == "__main__":
    unittest.main()
