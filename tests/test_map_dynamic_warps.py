import types
import unittest
from unittest.mock import patch


class TestMapWarpDynamicDestination(unittest.TestCase):
    def test_emerald_inside_truck_exit_uses_active_littleroot_destination(self):
        from modules.context import context
        from modules.map import MapWarp
        from modules.map_data import MapRSE
        from modules.profiles import Profile
        from modules.roms import ROM, ROMLanguage

        context.profile = Profile(
            ROM("", "", "POKEMON EMER", "", ROMLanguage.English, "", 0),
            "",
            None,
        )

        littleroot = types.SimpleNamespace(
            map_group_and_number=MapRSE.LITTLEROOT_TOWN.value,
            local_position=None,
            warps=[types.SimpleNamespace(local_coordinates=(10, 10))],
        )
        for y in (1, 2, 3):
            with self.subTest(exit_coordinates=(4, y)):
                truck_exit = MapWarp(bytes((4, 0, y, 0, 0, 0, 127, 127)))

                # SaveBlock1.dynamicWarp is a WarpData: x, y, elevation,
                # warp_id, map_number, map_group.  The 127/127 ROM fields
                # select this record; they are not a map ID or warp ID.
                dynamic_warp = bytes((20, 0, 16, 0, 0, 0xFF, 9, 0))
                with (
                    patch("modules.map.get_save_block", return_value=dynamic_warp),
                    patch("modules.map.get_map_data", return_value=littleroot),
                ):
                    destination = truck_exit.destination_location

                self.assertEqual(destination.map_group_and_number, MapRSE.LITTLEROOT_TOWN.value)


if __name__ == "__main__":
    unittest.main()
