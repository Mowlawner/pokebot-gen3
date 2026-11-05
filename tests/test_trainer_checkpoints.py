from types import SimpleNamespace
from unittest.mock import patch

from modules.map_data import MapRSE
from modules.trainer_checkpoints import (
    PETALBURG_WOODS_AQUA_GRUNT,
    ROUTE103_INTRO_RIVAL,
    ROUTE102_FIRST_POOCHYENA,
    RUSTURF_TUNNEL_AQUA_GRUNT,
    known_moves_for_battle,
    trainer_checkpoint_for_battle,
    trainer_checkpoint_for_id,
    trainer_checkpoint_for_source,
)
from modules.trainer_catalog import CatalogPokemon, CatalogTrainer


def test_route102_first_poochyena_checkpoint_is_version_specific_and_known():
    assert ROUTE102_FIRST_POOCHYENA.map_id == MapRSE.ROUTE102.value
    assert ROUTE102_FIRST_POOCHYENA.trainer_affordance_id == f"trainer:{MapRSE.ROUTE102.value}:2"
    assert ROUTE102_FIRST_POOCHYENA.party[0].species == "Poochyena"
    assert ROUTE102_FIRST_POOCHYENA.party[0].level == 5
    assert ROUTE102_FIRST_POOCHYENA.party[0].moves == ("Tackle", "Howl")
    assert trainer_checkpoint_for_id(ROUTE102_FIRST_POOCHYENA.trainer_affordance_id) is ROUTE102_FIRST_POOCHYENA


def test_route102_checkpoint_can_match_a_battle_by_map_species_and_level():
    opponent = SimpleNamespace(species=SimpleNamespace(name="Poochyena"), level=5)
    with patch(
        "modules.player.get_player_avatar",
        return_value=SimpleNamespace(map_group_and_number=MapRSE.ROUTE102.value),
    ):
        checkpoint = trainer_checkpoint_for_battle(SimpleNamespace(is_trainer_battle=True), opponent)
        moves = known_moves_for_battle(SimpleNamespace(is_trainer_battle=True), opponent)

    assert checkpoint is ROUTE102_FIRST_POOCHYENA
    assert moves == ("Tackle", "Howl")


def test_intro_rival_checkpoint_resolves_the_active_gender_script_roster():
    may = CatalogTrainer(
        trainer_id=21,
        name="May",
        trainer_class=1,
        encounter_music=0,
        gender=1,
        party_flags=2,
        ai_flags=0,
        party=(CatalogPokemon(1, 5, moves=(33, 45, 0, 0)),),
    )
    brendan = CatalogTrainer(
        trainer_id=22,
        name="Brendan",
        trainer_class=1,
        encounter_music=0,
        gender=0,
        party_flags=2,
        ai_flags=0,
        party=(CatalogPokemon(2, 5, moves=(33, 45, 0, 0)),),
    )
    catalog = (None,) * 21 + (may, brendan)
    with (
        patch("modules.trainer_catalog.active_trainer_catalog", return_value=catalog),
        patch(
            "modules.memory.read_symbol",
            side_effect=lambda name, **kwargs: b"\x5c\x15\x00" if name.endswith("RivalMay") else b"\x5c\x16\x00",
        ),
        patch("modules.player.get_player", return_value=SimpleNamespace(gender="male")),
        patch("modules.pokemon.get_species_by_index", side_effect=lambda index: SimpleNamespace(name=f"Species{index}")),
        patch(
            "modules.pokemon.get_move_by_index",
            side_effect=lambda index: SimpleNamespace(name={33: "Tackle", 45: "Growl"}[index]),
        ),
    ):
        checkpoint = trainer_checkpoint_for_source(ROUTE103_INTRO_RIVAL.trainer_affordance_id)

    assert checkpoint is not None
    assert checkpoint.source == "active_rom"
    assert checkpoint.rom_trainer_id == 21
    assert checkpoint.party[0].species == "Species1"


def test_source_resolution_reads_long_ordinary_trainer_wrappers():
    trainer = CatalogTrainer(
        trainer_id=21,
        name="Lady Cindy",
        trainer_class=1,
        encounter_music=0,
        gender=1,
        party_flags=2,
        ai_flags=0,
        party=(CatalogPokemon(1, 7, moves=(33, 45, 0, 0)),),
    )
    catalog = (None,) * 21 + (trainer,)
    calls = []

    def read_script(_name, **kwargs):
        calls.append(kwargs)
        return b"\x01" * 80 + b"\x5c\x15\x00"

    with (
        patch("modules.trainer_catalog.active_trainer_catalog", return_value=catalog),
        patch("modules.memory.read_symbol", side_effect=read_script),
        patch("modules.pokemon.get_species_by_index", return_value=SimpleNamespace(name="Poochyena")),
        patch(
            "modules.pokemon.get_move_by_index",
            side_effect=lambda index: SimpleNamespace(name={33: "Tackle", 45: "Growl"}[index]),
        ),
    ):
        checkpoint = trainer_checkpoint_for_source(
            "trainer:(0, 17):7",
            script_symbol="Route102_EventScript_LadyCindy",
        )

    assert checkpoint is not None
    assert checkpoint.display_name == "Lady Cindy"
    assert calls and calls[0]["size"] >= 128


def test_script_owned_story_checkpoints_resolve_the_active_rom_roster():
    trainer = CatalogTrainer(
        trainer_id=21,
        name="Aqua Grunt",
        trainer_class=1,
        encounter_music=0,
        gender=0,
        party_flags=2,
        ai_flags=0,
        party=(CatalogPokemon(1, 9, moves=(33, 45, 0, 0)),),
    )
    catalog = (None,) * 21 + (trainer,)

    with (
        patch("modules.trainer_catalog.active_trainer_catalog", return_value=catalog),
        patch("modules.memory.read_symbol", return_value=b"\x5c\x15\x00"),
        patch("modules.pokemon.get_species_by_index", return_value=SimpleNamespace(name="Poochyena")),
        patch(
            "modules.pokemon.get_move_by_index",
            side_effect=lambda index: SimpleNamespace(name={33: "Tackle", 45: "Growl"}[index]),
        ),
    ):
        woods = trainer_checkpoint_for_source(
            PETALBURG_WOODS_AQUA_GRUNT.trainer_affordance_id,
            script_symbol="PetalburgWoods_EventScript_DevonResearcherLeft",
        )
        tunnel = trainer_checkpoint_for_source(
            RUSTURF_TUNNEL_AQUA_GRUNT.trainer_affordance_id,
            script_symbol="RusturfTunnel_EventScript_Grunt",
        )

    for checkpoint, expected_id in (
        (woods, PETALBURG_WOODS_AQUA_GRUNT.checkpoint_id),
        (tunnel, RUSTURF_TUNNEL_AQUA_GRUNT.checkpoint_id),
    ):
        assert checkpoint is not None
        assert checkpoint.checkpoint_id == expected_id
        assert checkpoint.source == "active_rom"
        assert checkpoint.rom_trainer_id == 21
        assert checkpoint.party[0].species == "Poochyena"
        assert checkpoint.party[0].moves == ("Tackle", "Growl")


def test_intro_rival_checkpoint_uses_starter_context_script_when_gender_branch_is_wrapper():
    may_treecko = CatalogTrainer(
        trainer_id=0x314,
        name="May",
        trainer_class=1,
        encounter_music=0,
        gender=1,
        party_flags=2,
        ai_flags=0,
        party=(CatalogPokemon(1, 5, moves=(33, 45, 0, 0)),),
    )
    catalog = (None,) * 0x314 + (may_treecko,)

    def read_script(name, **_kwargs):
        if name == "Route103_EventScript_StartMayBattleTreecko":
            return b"\x5c\x14\x03"
        if name == "Route103_EventScript_RivalMay":
            return b"\x02\x0f\x00"
        return b"\x00\x00\x00"

    with (
        patch("modules.trainer_catalog.active_trainer_catalog", return_value=catalog),
        patch("modules.memory.read_symbol", side_effect=read_script),
        patch("modules.player.get_player", return_value=SimpleNamespace(gender="male")),
        patch(
            "modules.pokemon_party.get_party",
            return_value=(SimpleNamespace(is_egg=False, species=SimpleNamespace(name="Treecko")),),
        ),
        patch("modules.pokemon.get_species_by_index", return_value=SimpleNamespace(name="Treecko")),
        patch(
            "modules.pokemon.get_move_by_index",
            side_effect=lambda index: SimpleNamespace(name={33: "Tackle", 45: "Growl"}[index]),
        ),
    ):
        checkpoint = trainer_checkpoint_for_source(ROUTE103_INTRO_RIVAL.trainer_affordance_id)

    assert checkpoint is not None
    assert checkpoint.rom_trainer_id == 0x314
    assert checkpoint.party[0].species == "Treecko"


def test_live_battle_uses_rom_trainer_catalog_before_static_registry():
    record = CatalogTrainer(
        trainer_id=12,
        name="Leader Roxanne",
        trainer_class=1,
        encounter_music=0,
        gender=0,
        party_flags=2,
        ai_flags=0,
        party=(CatalogPokemon(1, 10, moves=(33, 45, 0, 0)),),
    )
    opponent = SimpleNamespace(
        species=SimpleNamespace(name="Geodude"),
        level=10,
        party_index=0,
    )
    with (
        patch("modules.trainer_catalog.active_trainer_catalog", return_value=(None,) * 12 + (record,)),
        patch("modules.memory.read_symbol", return_value=b"\x0c\x00"),
        patch("modules.pokemon.get_species_by_index", return_value=SimpleNamespace(name="Geodude")),
        patch(
            "modules.pokemon.get_move_by_index",
            side_effect=lambda index: SimpleNamespace(name={33: "Tackle", 45: "Rock Tomb"}[index]),
        ),
        patch(
            "modules.player.get_player_avatar",
            return_value=SimpleNamespace(map_group_and_number=MapRSE.ROUTE116.value),
        ),
    ):
        checkpoint = trainer_checkpoint_for_battle(SimpleNamespace(is_trainer_battle=True), opponent)
        moves = known_moves_for_battle(SimpleNamespace(is_trainer_battle=True), opponent)

    assert checkpoint is not None
    assert checkpoint.source == "active_rom"
    assert checkpoint.rom_trainer_id == 12
    assert checkpoint.party[0].moves == ("Tackle", "Rock Tomb")
    assert moves == ("Tackle", "Rock Tomb")
