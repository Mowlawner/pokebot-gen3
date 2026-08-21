import unittest
from unittest.mock import patch

from modules.items import InvalidItemIndexError
from modules.nuzlocke.resource_policy import ResourceObservationStatus
from modules.nuzlocke.resource_runtime import observe_resource_snapshot


class _Pokemon:
    current_hp = 10
    total_hp = 20
    status_condition = type("Status", (), {"value": "none"})()


class _Bag:
    items = ()


class _Storage:
    @property
    def items(self):
        raise InvalidItemIndexError(65535, slot=3, storage="pc")


class ResourceObservationBoundaryTests(unittest.TestCase):
    def test_unavailable_bag_is_not_empty(self):
        with patch("modules.nuzlocke.resource_runtime.get_party", return_value=(_Pokemon(),)), patch(
            "modules.nuzlocke.resource_runtime.get_item_bag", return_value=None
        ), patch("modules.nuzlocke.resource_runtime.get_item_storage", return_value=type("Storage", (), {})()):
            result = observe_resource_snapshot()
        self.assertEqual(result.observation_status, ResourceObservationStatus.UNAVAILABLE)
        self.assertEqual(result.observation_error, "item_bag_unavailable")

    def test_unavailable_storage_is_not_empty(self):
        with patch("modules.nuzlocke.resource_runtime.get_party", return_value=(_Pokemon(),)), patch(
            "modules.nuzlocke.resource_runtime.get_item_bag", return_value=_Bag()
        ), patch("modules.nuzlocke.resource_runtime.get_item_storage", return_value=None):
            result = observe_resource_snapshot()
        self.assertEqual(result.observation_status, ResourceObservationStatus.UNAVAILABLE)
        self.assertEqual(result.observation_error, "item_storage_unavailable")

    def test_unavailable_party_is_not_empty(self):
        with patch("modules.nuzlocke.resource_runtime.get_party", return_value=None), patch(
            "modules.nuzlocke.resource_runtime.get_item_bag", return_value=_Bag()
        ), patch("modules.nuzlocke.resource_runtime.get_item_storage", return_value=_Storage()):
            result = observe_resource_snapshot()
        self.assertEqual(result.observation_status, ResourceObservationStatus.UNAVAILABLE)
        self.assertEqual(result.unavailable_components, ("party",))

    def test_party_and_bag_unavailable_are_both_reported(self):
        with patch("modules.nuzlocke.resource_runtime.get_party", return_value=None), patch(
            "modules.nuzlocke.resource_runtime.get_item_bag", return_value=None
        ), patch("modules.nuzlocke.resource_runtime.get_item_storage", return_value=None):
            result = observe_resource_snapshot()
        self.assertEqual(result.unavailable_components, ("party", "item_bag", "item_storage"))

    def test_unexpected_reader_error_propagates(self):
        with patch("modules.nuzlocke.resource_runtime.get_party", return_value=(_Pokemon(),)), patch(
            "modules.nuzlocke.resource_runtime.get_item_bag", side_effect=RuntimeError("boom")
        ):
            with self.assertRaises(RuntimeError):
                observe_resource_snapshot()

    def test_valid_empty_storage_is_not_malformed(self):
        storage = type("Storage", (), {"items": ()})()
        with patch("modules.nuzlocke.resource_runtime.get_party", return_value=(_Pokemon(),)), patch(
            "modules.nuzlocke.resource_runtime.get_item_bag", return_value=_Bag()
        ), patch("modules.nuzlocke.resource_runtime.get_item_storage", return_value=storage):
            result = observe_resource_snapshot()
        self.assertEqual(result.observation_status, ResourceObservationStatus.VALID)
        self.assertEqual(result.pc_healing_items, ())

    def test_invalid_index_is_malformed_not_empty(self):
        with patch("modules.nuzlocke.resource_runtime.get_party", return_value=(_Pokemon(),)), patch(
            "modules.nuzlocke.resource_runtime.get_item_bag", return_value=_Bag()
        ), patch("modules.nuzlocke.resource_runtime.get_item_storage", return_value=_Storage()):
            result = observe_resource_snapshot()
        self.assertEqual(result.observation_status, ResourceObservationStatus.MALFORMED)
        self.assertEqual(result.invalid_item_index, 65535)
        self.assertEqual(result.invalid_item_slot, 3)
        self.assertEqual(result.invalid_item_storage, "pc")
        self.assertEqual(result.pc_healing_items, ())


if __name__ == "__main__":
    unittest.main()
