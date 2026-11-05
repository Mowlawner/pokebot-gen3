from types import SimpleNamespace

from modules.battle_move_policy import choose_move_replacement, move_policy_summary


def move(name, *, power=0, accuracy=1.0, effect="HIT", type_name="Normal", secondary_accuracy=0.0, tm_hm=None):
    return SimpleNamespace(
        name=name,
        base_power=power,
        accuracy=accuracy,
        effect=effect,
        secondary_accuracy=secondary_accuracy,
        type=SimpleNamespace(name=type_name),
        pp=20,
        tm_hm=tm_hm,
    )


def pokemon(moves, *, species_types=("Normal",)):
    return SimpleNamespace(
        moves=tuple(SimpleNamespace(move=item) if item is not None else None for item in moves),
        species=SimpleNamespace(types=tuple(SimpleNamespace(name=name) for name in species_types)),
    )


def test_move_policy_fills_an_empty_slot():
    result = choose_move_replacement(pokemon((move("Tackle", power=35), None, None, None)), move("Howl"))

    assert result.replacement_index == 1
    assert "empty move slot" in result.reason


def test_move_policy_treats_damaging_status_move_as_damage_first():
    result = choose_move_replacement(
        pokemon((move("Tackle", power=35), move("Scratch", power=40), move("Growl"), move("Tail Whip"))),
        move("Smog", power=20, effect="POISON", secondary_accuracy=0.4),
    )

    assert result.replacement_index in (0, 2, 3)
    assert move_policy_summary(move("Smog", power=20, effect="POISON")) == "damage_over_time_unsafe_for_capture"


def test_move_policy_does_not_replace_a_weak_move_with_damage_over_time_move():
    result = choose_move_replacement(
        pokemon((move("Tackle", power=35), move("Scratch", power=40), move("Growl"), move("Tail Whip"))),
        move("Toxic", effect="TOXIC", accuracy=0.85),
    )

    assert result.replacement_index == 4


def test_move_policy_preserves_hm_slots():
    hm_cut = move("Cut", power=50, tm_hm=SimpleNamespace(name="HM01 Cut"))
    result = choose_move_replacement(
        pokemon((hm_cut, move("Growl"), move("Tail Whip"), move("Tackle", power=35))),
        move("Flamethrower", power=95, type_name="Fire"),
    )

    assert result.replacement_index != 0
