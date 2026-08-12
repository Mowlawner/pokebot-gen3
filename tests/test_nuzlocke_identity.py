import unittest


class TestPokemonIdentity(unittest.TestCase):
    def test_persistent_fields_form_immutable_hashable_identity(self):
        from modules.nuzlocke.identity import PokemonIdentity

        identity = PokemonIdentity(123, 456, 789)
        self.assertEqual(identity, PokemonIdentity(123, 456, 789))
        self.assertEqual(len({identity}), 1)
        with self.assertRaises(AttributeError):
            identity.personality_value = 1

    def test_validity_is_required_for_model_identity(self):
        from modules.nuzlocke.identity import PokemonIdentity

        class FakePokemon:
            is_empty = False
            is_valid = True
            personality_value = 10
            original_trainer = type("Trainer", (), {"id": 20, "secret_id": 30})()

        self.assertEqual(PokemonIdentity.from_pokemon(FakePokemon()), PokemonIdentity(10, 20, 30))
        FakePokemon.is_valid = False
        self.assertIsNone(PokemonIdentity.from_pokemon(FakePokemon()))
        FakePokemon.is_valid = True
        FakePokemon.is_empty = True
        self.assertIsNone(PokemonIdentity.from_pokemon(FakePokemon()))

    def test_battle_identity_matches_party_identity(self):
        from modules.nuzlocke.identity import PokemonIdentity

        class FakeBattlePokemon:
            personality_value = 10
            ot_id = (30 << 16) | 20

        self.assertEqual(
            PokemonIdentity.from_battle_pokemon(FakeBattlePokemon()),
            PokemonIdentity(10, 20, 30),
        )

    def test_identity_survives_location_and_mutable_snapshot_changes(self):
        from modules.nuzlocke.identity import PokemonIdentity
        from modules.nuzlocke.snapshots import PartyPokemonSnapshot, StoragePokemonSnapshot

        identity = PokemonIdentity(10, 20, 30)
        party = PartyPokemonSnapshot(party_index=0, species="Treecko", nickname="Sprig", level=5,
                                     current_hp=10, max_hp=10, status="none", personality_value=10,
                                     original_trainer_id=20, original_trainer_secret_id=30,
                                     original_trainer_name="May", moves=(), held_item=None,
                                     fainted=False, egg=False, identity=identity)
        pc = StoragePokemonSnapshot(3, 14, PartyPokemonSnapshot(
            party_index=5, species="Grovyle", nickname="Sprig", level=16, current_hp=20,
            max_hp=30, status="poison", personality_value=10, original_trainer_id=20,
            original_trainer_secret_id=30, original_trainer_name="May", moves=(), held_item=None,
            fainted=False, egg=False, identity=identity))
        self.assertEqual(party.identity, pc.pokemon.identity)

    def test_invalid_and_empty_slots_have_no_identity(self):
        from modules.nuzlocke.identity import PokemonIdentity
        from modules.nuzlocke.snapshots import StorageSnapshot

        self.assertEqual(StorageSnapshot(0, ()).pokemon, ())
        invalid = type("Invalid", (), {"is_empty": False, "is_valid": False})()
        empty = type("Empty", (), {"is_empty": True, "is_valid": False})()
        self.assertIsNone(PokemonIdentity.from_pokemon(invalid))
        self.assertIsNone(PokemonIdentity.from_pokemon(empty))

    def test_storage_move_event_and_projection_follow_identity(self):
        from modules.nuzlocke.events import PokemonStorageLocation, StorageChanged
        from modules.nuzlocke.identity import PokemonIdentity
        from modules.nuzlocke.projection import CampaignProjection

        identity = PokemonIdentity(1, 2, 3)
        event = StorageChanged(
            2,
            (),
            (),
            ((PokemonStorageLocation(identity, 0, 1), PokemonStorageLocation(identity, 4, 29)),),
        )
        projection = CampaignProjection()
        projection.apply(StorageChanged(1, (PokemonStorageLocation(identity, 0, 1),), ()))
        projection.apply(event)
        self.assertEqual(projection.state.pc_locations, ((identity, 4, 29),))


if __name__ == "__main__":
    unittest.main()
