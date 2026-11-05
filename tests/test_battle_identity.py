from types import SimpleNamespace

from modules.battle_identity import battle_pokemon_display_name, battle_pokemon_identity


def test_battle_identity_keeps_species_and_custom_nickname_together():
    party_member = SimpleNamespace(
        index=3,
        species=SimpleNamespace(name="Poochyena"),
        nickname="Jennifer",
    )
    battler = SimpleNamespace(
        party_index=3,
        species=SimpleNamespace(name="Poochyena"),
    )

    class Party:
        def __getitem__(self, index):
            if index == 3:
                return party_member
            raise IndexError(index)

    party = Party()
    assert battle_pokemon_identity(battler, party=party) == ("Poochyena", "Jennifer", 3)
    assert battle_pokemon_display_name(battler, party=party) == 'Poochyena "Jennifer"'
