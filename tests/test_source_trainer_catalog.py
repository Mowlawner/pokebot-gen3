from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from modules.nuzlocke.source_trainer_catalog import parse_source_trainer_catalog, trainer_metadata


def _source_fixture(root: Path) -> None:
    (root / "include" / "constants").mkdir(parents=True)
    (root / "src" / "data").mkdir(parents=True)
    (root / "include" / "constants" / "opponents.h").write_text(
        "#define TRAINER_FIXTURE 318\n#define TRAINER_EMPTY 319\n", encoding="utf-8"
    )
    (root / "src" / "data" / "trainer_parties.h").write_text(
        """static const struct TrainerMonNoItemDefaultMoves sParty_Fixture[] = {
    {
    .iv = 0,
    .lvl = 5,
    .species = SPECIES_POOCHYENA,
    }
};
""",
        encoding="utf-8",
    )
    (root / "src" / "data" / "trainers.h").write_text(
        """const struct Trainer gTrainers[] = {
    [TRAINER_FIXTURE] =
    {
        .trainerName = _(\"FIXTURE\"),
        .doubleBattle = FALSE,
        .party = NO_ITEM_DEFAULT_MOVES(sParty_Fixture),
    },
    [TRAINER_EMPTY] =
    {
        .trainerName = _(\"EMPTY\"),
        .doubleBattle = TRUE,
        .party = NO_ITEM_DEFAULT_MOVES(sParty_Missing),
    },
};
""",
        encoding="utf-8",
    )


class SourceTrainerCatalogTests(unittest.TestCase):
    def test_parses_identity_party_and_provenance(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _source_fixture(root)
            records = parse_source_trainer_catalog(root)
        fixture = next(record for record in records if record.trainer_symbol == "TRAINER_FIXTURE")
        self.assertEqual(fixture.trainer_id, 318)
        self.assertEqual(fixture.display_name, "FIXTURE")
        self.assertEqual(fixture.party[0].species, "SPECIES_POOCHYENA")
        self.assertEqual(fixture.party[0].level, "5")
        self.assertEqual(fixture.source.symbol, "TRAINER_FIXTURE")
        self.assertEqual(fixture.party_source.symbol, "sParty_Fixture")
        metadata = trainer_metadata(fixture)
        self.assertEqual(metadata["roster_resolution"], "resolved")

    def test_keeps_unresolved_party_explicit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _source_fixture(root)
            records = parse_source_trainer_catalog(root)
        empty = next(record for record in records if record.trainer_symbol == "TRAINER_EMPTY")
        self.assertEqual(empty.party, ())
        self.assertEqual(trainer_metadata(empty)["roster_resolution"], "unresolved")
        self.assertEqual(trainer_metadata(empty)["double_battle"], "true")


if __name__ == "__main__":
    unittest.main()
