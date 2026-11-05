from types import SimpleNamespace
from unittest.mock import patch

from modules.modes._listeners import BattleListener
from modules.nuzlocke.held_item_policy import evaluate_held_item, party_held_items_needing_removal


def _pokemon(item, ability="None"):
    return SimpleNamespace(
        name="test-mon",
        held_item=item,
        ability=SimpleNamespace(name=ability),
        is_egg=False,
    )


def test_keep_items_with_a_real_hold_effect_for_any_ability():
    item = SimpleNamespace(name="Oran Berry", hold_effect=SimpleNamespace(name="RestoreHP"))

    decision = evaluate_held_item(_pokemon(item))

    assert decision.retain is True


def test_remove_inert_wild_item_even_when_the_pokemon_has_no_pickup():
    item = SimpleNamespace(name="Mail", hold_effect=SimpleNamespace(name="NoEffect"))

    decision = evaluate_held_item(_pokemon(item, ability="Overgrow"))

    assert decision.retain is False
    assert "no strategic hold effect" in decision.reason


def test_remove_inert_pickup_result_and_keep_explicitly_reserved_item():
    item = SimpleNamespace(name="Potion", hold_effect=SimpleNamespace(name="NoEffect"))
    reserved = evaluate_held_item(_pokemon(item, ability="Pickup"), planned_item_names=("Potion",))

    assert reserved.retain is True
    assert party_held_items_needing_removal((_pokemon(item, ability="Pickup"),)) == (0,)


def test_collect_pickup_result_even_when_the_item_has_a_hold_effect():
    item = SimpleNamespace(name="King's Rock", hold_effect=SimpleNamespace(name="Flinch"))

    decision = evaluate_held_item(_pokemon(item, ability="Pickup"))

    assert decision.retain is False
    assert "Pickup result" in decision.reason


def test_retrieve_held_items_matches_pickup_name_punctuation_variants():
    listener = BattleListener()
    item = SimpleNamespace(name="King's Rock", hold_effect=SimpleNamespace(name="Flinch"))
    pokemon = _pokemon(item, ability="Pickup")
    pokemon.index = 0
    result = SimpleNamespace(party_indices_with_stolen_items=[], party_indices_with_picked_up_items=[])

    def return_items(_indices, force_checking=False):
        assert force_checking is True
        if False:
            yield
        return [item]

    with (
        patch("modules.modes._listeners.get_items_available_for_pickup", return_value=["King’s Rock"]),
        patch("modules.modes._listeners.get_party", return_value=[pokemon]),
        patch(
            "modules.modes._listeners.take_held_items_from_multiple_party_pokemon",
            side_effect=return_items,
        ) as take,
        patch("modules.modes._listeners.plugin_picked_up_items", return_value=iter(())),
        patch("modules.modes._listeners.diagnostic_print"),
    ):
        list(listener.retrieve_held_items(result))

    take.assert_called_once_with([0, 0], force_checking=True)
