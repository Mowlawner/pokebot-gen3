"""Known early-game trainer checkpoints for campaign battle planning.

The registry is intentionally small and version-specific.  It is a seed for
the broader trainer-data adapter: unknown trainers remain unknown instead of
being treated as if their party were empty.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from modules.console import diagnostic_print
from modules.map_data import MapRSE


_last_source_resolution_probe: tuple | None = None
_source_checkpoint_cache: dict[tuple, "TrainerCheckpoint | None"] = {}

# Object-event scripts frequently contain lock/face/call/setup commands before
# their trainerbattle command.  Keep the scan bounded, but make the window
# large enough for ordinary field wrappers as well as the short story scripts.
# The resolver still accepts a trainer ID only when it is valid for the active
# catalog, so this does not turn the scan into arbitrary ROM interpretation.
_TRAINER_SCRIPT_SCAN_BYTES = 256


def _source_resolution_facts(checkpoint: "TrainerCheckpoint | None") -> tuple[str | None, str | None, tuple[str, ...]]:
    """Read the small set of live facts needed to choose a scripted roster.

    This is kept separate from resolution so the entry diagnostic records the
    facts available at the boundary where a checkpoint was requested.  The
    probe is deduplicated below; repeated readiness polls therefore do not
    become a per-frame console/logging cost.
    """

    gender = None
    starter = None
    symbols: list[str] = []
    if checkpoint is None:
        return gender, starter, ()
    # Fixed story trainers do not need save-backed context.  Keeping this
    # branch lazy is important during script transitions and in ROM fixtures:
    # the caller may have a script reader but no valid save-block reader yet.
    # Only context-selected scripts, such as the introductory rival branch,
    # require the player's gender and starter species.
    context_symbols = dict(checkpoint.battle_script_symbols_by_context)
    if context_symbols:
        try:
            from modules.player import get_player

            gender = get_player().gender
        except (AttributeError, RuntimeError, TypeError, ValueError, IndexError):
            pass
        try:
            from modules.pokemon_party import get_party

            starter = next(
                (
                    getattr(getattr(pokemon, "species", None), "name", None)
                    for pokemon in get_party()
                    if not getattr(pokemon, "is_egg", False)
                    and getattr(getattr(pokemon, "species", None), "name", None)
                ),
                None,
            )
        except (AttributeError, RuntimeError, TypeError, ValueError, IndexError):
            pass
        context_symbol = context_symbols.get(f"{gender}:{starter}")
        if context_symbol:
            symbols.append(context_symbol)
    if checkpoint.script_symbol:
        symbols.append(checkpoint.script_symbol)
    symbols.extend(checkpoint.alternate_script_symbols)
    return gender, starter, tuple(dict.fromkeys(symbols))


def _source_resolution_probe(
    trainer_affordance_id: str | None,
    script_symbol: str | None,
    checkpoint: "TrainerCheckpoint | None",
    *,
    catalog_size: int | None,
    gender: str | None,
    starter: str | None,
    symbols: tuple[str, ...],
) -> None:
    global _last_source_resolution_probe
    probe = (
        trainer_affordance_id,
        script_symbol,
        getattr(checkpoint, "checkpoint_id", None),
        catalog_size,
        gender,
        starter,
        symbols,
    )
    if probe == _last_source_resolution_probe:
        return
    _last_source_resolution_probe = probe
    diagnostic_print(
        lambda: (
            "TRAINER_CHECKPOINT_RESOLUTION: "
            f"phase='entry' affordance={trainer_affordance_id!r} "
            f"script_symbol={script_symbol!r} catalog_size={catalog_size!r} "
            f"player_gender={gender!r} starter_species={starter!r} "
            f"candidate_symbols={symbols!r}"
        ),
        trace=True,
        prefix="TRAINER_CHECKPOINT_RESOLUTION",
    )


@dataclass(frozen=True, slots=True)
class TrainerPokemon:
    species: str
    level: int
    moves: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class TrainerCheckpoint:
    checkpoint_id: str
    map_id: tuple[int, int]
    trainer_affordance_id: str
    display_name: str
    party: tuple[TrainerPokemon, ...]
    mandatory: bool = False
    source: str = "verified_game_data"
    rom_trainer_id: int | None = None
    trainer_class: int | None = None
    ai_flags: int | None = None
    gym_trainer: bool = False
    # A campaign fact that proves the checkpoint is behind the player.  This
    # is intentionally separate from the live object flag: the object may not
    # be present after a map transition, while the campaign milestone remains
    # authoritative for route planning.
    clearance_fact: str | None = None
    # Some story trainers select their concrete trainer ID through a ROM
    # script (for example, Emerald's Route 103 rival chooses a script for the
    # player's gender).  Keep those script symbols with the checkpoint so the
    # registry can resolve the active ROM roster without guessing a party.
    script_symbol: str | None = None
    alternate_script_symbols: tuple[str, ...] = ()
    # Some scripts choose a concrete trainer from more than one ROM branch.
    # The key is deliberately declarative (for example ``male:Mudkip``), so
    # the source resolver can use live save facts without putting a rival
    # special case in campaign execution.
    battle_script_symbols_by_context: tuple[tuple[str, str], ...] = ()
    # When ROM-backed and verified route facts describe the same trainer, keep
    # every source ID for diagnostics while exposing one logical checkpoint to
    # campaign policy.
    source_records: tuple[tuple[str, str], ...] = ()
    # Source-derived checkpoints retain the gate candidate that authorized
    # their promotion.  These fields are optional for the existing verified
    # registry and active-ROM catalog records.
    candidate_id: str | None = None
    provenance: tuple[tuple[str, str], ...] = ()
    confidence: str = "verified_registry"
    verification_status: str = "verified"
    # ``story`` and ``mandatory_battle`` checkpoints own campaign
    # preparation. ``strategic`` checkpoints are promoted by policy, while
    # ordinary hazards remain route-exposure decisions.
    checkpoint_kind: str = "hazard"


def trainer_checkpoint_identity(checkpoint) -> tuple:
    """Return the stable logical identity shared by trainer fact sources."""

    map_id = getattr(getattr(checkpoint, "map_id", None), "value", getattr(checkpoint, "map_id", None))
    affordance_id = getattr(checkpoint, "trainer_affordance_id", None)
    if affordance_id:
        return ("affordance", map_id, affordance_id)
    return (
        "display",
        map_id,
        str(getattr(checkpoint, "display_name", "") or "").casefold(),
    )


def trainer_checkpoint_sources(checkpoint) -> tuple[tuple[str, str], ...]:
    """Return source/provenance pairs, including the checkpoint's own ID."""

    records = tuple(getattr(checkpoint, "source_records", ()) or ())
    if records:
        return records
    source = str(getattr(checkpoint, "source", "unknown") or "unknown")
    checkpoint_id = str(getattr(checkpoint, "checkpoint_id", "unknown") or "unknown")
    return ((source, checkpoint_id),)


def merge_trainer_checkpoints(*checkpoints):
    """Merge equivalent facts while retaining the richest party data."""

    candidates = tuple(checkpoint for checkpoint in checkpoints if checkpoint is not None)
    if not candidates:
        return None
    primary = max(
        candidates,
        key=lambda checkpoint: (
            bool(getattr(checkpoint, "party", ())),
            getattr(checkpoint, "source", None) == "active_rom",
            len(getattr(checkpoint, "party", ()) or ()),
        ),
    )
    records = []
    for checkpoint in candidates:
        for record in trainer_checkpoint_sources(checkpoint):
            if record not in records:
                records.append(record)
    try:
        return replace(primary, source_records=tuple(records))
    except TypeError:
        # Lightweight campaign tests and adapters sometimes use a mutable
        # namespace in place of the dataclass. Preserve that seam while still
        # making the merged provenance available to the snapshot renderer.
        try:
            primary.source_records = tuple(records)
        except (AttributeError, TypeError):
            pass
        return primary


# Youngster Calvin's first Route 102 battle in Emerald/Ruby/Sapphire.  In the
# Route 102 object table Calvin is local ID 2 (local ID 1 is not Calvin), so
# keep the affordance identity aligned with the ROM object that creates the
# unavoidable encounter. The level-5 Poochyena has the moves available
# through level 5 in the Gen III trainer data: Tackle and Howl.
ROUTE102_FIRST_POOCHYENA = TrainerCheckpoint(
    checkpoint_id="route102:first_poochyena",
    map_id=MapRSE.ROUTE102.value,
    trainer_affordance_id=f"trainer:{MapRSE.ROUTE102.value}:2",
    display_name="Calvin",
    party=(TrainerPokemon("Poochyena", 5, ("Tackle", "Howl")),),
    mandatory=True,
    checkpoint_kind="mandatory_battle",
    # Reaching Petalburg proves the player crossed the first unavoidable
    # Route 102 trainer.  Until that ROM-backed fact is observed, routes beyond
    # Route 102 remain preparation routes.
    clearance_fact="visited_petalburg",
)


# The introductory Route 103 rival is a mandatory story battle.  Its concrete
# trainer ID and party are selected by the gender branch in the Emerald event
# script, so the empty party is deliberate: trainer_checkpoint_for_source()
# resolves the active ROM record from the branch-specific script symbols.
ROUTE103_INTRO_RIVAL = TrainerCheckpoint(
    checkpoint_id="route103:intro_rival",
    map_id=MapRSE.ROUTE103.value,
    trainer_affordance_id=f"trainer:{MapRSE.ROUTE103.value}:2",
    display_name="Introductory Rival",
    party=(),
    mandatory=True,
    checkpoint_kind="story",
    source="rom_script",
    clearance_fact="intro_rival_battle_complete",
    script_symbol="Route103_EventScript_RivalMay",
    alternate_script_symbols=("Route103_EventScript_RivalBrendan",),
    battle_script_symbols_by_context=(
        # The suffix identifies the player's starter branch. The rival's
        # concrete starter is the opposing type selected by that branch.
        ("male:Treecko", "Route103_EventScript_StartMayBattleTreecko"),
        ("male:Torchic", "Route103_EventScript_StartMayBattleTorchic"),
        ("male:Mudkip", "Route103_EventScript_StartMayBattleMudkip"),
        ("female:Treecko", "Route103_EventScript_StartBrendanBattleTreecko"),
        ("female:Torchic", "Route103_EventScript_StartBrendanBattleTorchic"),
        ("female:Mudkip", "Route103_EventScript_StartBrendanBattleMudkip"),
    ),
)


# Petalburg Woods starts with a researcher scene that hands off to the Aqua
# grunt battle.  The field interaction is the stable affordance; the nested
# intro script owns the concrete trainerbattle command and is resolved from
# the active ROM catalog below.
PETALBURG_WOODS_AQUA_GRUNT = TrainerCheckpoint(
    checkpoint_id="petalburg_woods:aqua_grunt",
    map_id=MapRSE.PETALBURG_WOODS.value,
    trainer_affordance_id="devon_goods_researcher",
    display_name="Petalburg Woods Aqua Grunt",
    party=(),
    mandatory=True,
    checkpoint_kind="story",
    source="rom_script",
    clearance_fact="petalburg_woods_scene_complete",
    script_symbol="PetalburgWoods_EventScript_DevonResearcherIntro",
    alternate_script_symbols=(
        "PetalburgWoods_EventScript_DevonResearcherLeft",
        "PetalburgWoods_EventScript_DevonResearcherRight",
    ),
)


# Rusturf Tunnel's Devon Goods scene is another script-owned trainer battle.
# Keep its story affordance separate from a map/local object ID because the
# object is moved and removed by the scene itself.
RUSTURF_TUNNEL_AQUA_GRUNT = TrainerCheckpoint(
    checkpoint_id="rusturf_tunnel:aqua_grunt",
    map_id=MapRSE.RUSTURF_TUNNEL.value,
    trainer_affordance_id="rusturf_tunnel_goods",
    display_name="Rusturf Tunnel Aqua Grunt",
    party=(),
    mandatory=True,
    checkpoint_kind="story",
    source="rom_script",
    clearance_fact="devon_goods_recovered",
    script_symbol="RusturfTunnel_EventScript_Grunt",
)


KNOWN_TRAINER_CHECKPOINTS: tuple[TrainerCheckpoint, ...] = (
    ROUTE102_FIRST_POOCHYENA,
    ROUTE103_INTRO_RIVAL,
    PETALBURG_WOODS_AQUA_GRUNT,
    RUSTURF_TUNNEL_AQUA_GRUNT,
)


def trainer_checkpoint_for_id(trainer_affordance_id: str | None) -> TrainerCheckpoint | None:
    if trainer_affordance_id is None:
        return None
    return next(
        (checkpoint for checkpoint in KNOWN_TRAINER_CHECKPOINTS if checkpoint.trainer_affordance_id == trainer_affordance_id),
        None,
    )


def trainer_checkpoint_for_battle(
    battle_state=None,
    opponent=None,
    *,
    trainer_affordance_id: str | None = None,
) -> TrainerCheckpoint | None:
    """Resolve the active trainer from the ROM catalog when possible.

    The overworld affordance is not available once a trainer battle is
    active, but the battle engine exposes ``gTrainerBattleOpponent_A``.  Use
    that stable ROM trainer ID first so battle planning and preparation share
    one roster source.  The small hand-verified registry remains the fallback
    for tests and ROM boundaries where the live trainer ID cannot be read.
    """

    exact = trainer_checkpoint_for_id(trainer_affordance_id)
    if exact is not None and exact.party:
        return exact
    if bool(getattr(battle_state, "is_trainer_battle", False)):
        rom_checkpoint = _trainer_checkpoint_for_live_battle()
        if rom_checkpoint is not None:
            return rom_checkpoint
    try:
        from modules.player import get_player_avatar

        map_id = getattr(get_player_avatar(), "map_group_and_number", None)
        map_id = getattr(map_id, "value", map_id)
    except (AttributeError, RuntimeError, TypeError, ValueError):
        map_id = None
    species = getattr(getattr(opponent, "species", None), "name", None)
    level = getattr(opponent, "level", None)
    candidates = tuple(checkpoint for checkpoint in KNOWN_TRAINER_CHECKPOINTS if checkpoint.map_id == map_id)
    matches = []
    for checkpoint in candidates:
        first = checkpoint.party[0] if checkpoint.party else None
        if first is not None and first.species == species and first.level == level:
            matches.append(checkpoint)
    # Species and level are useful fallbacks for battle observation, but they
    # are not a trainer identity. If a future registry contains two matching
    # profiles on the same map, refuse to guess and let the normal unknown
    # trainer policy take over.
    return matches[0] if len(matches) == 1 else exact if exact is not None and exact.party else None


def _map_id_from_affordance(trainer_affordance_id: str | None) -> tuple[int, int] | None:
    if not trainer_affordance_id:
        return None
    parts = trainer_affordance_id.split(":", 2)
    if len(parts) != 3:
        return None
    try:
        map_id = tuple(int(part.strip()) for part in parts[1].strip("() ").split(","))
        int(parts[2])
    except (TypeError, ValueError):
        return None
    return map_id if len(map_id) == 2 else None


def _current_map_id() -> tuple[int, int] | None:
    try:
        from modules.player import get_player_avatar

        map_id = getattr(get_player_avatar(), "map_group_and_number", None)
        map_id = getattr(map_id, "value", map_id)
        if isinstance(map_id, tuple) and len(map_id) == 2:
            return int(map_id[0]), int(map_id[1])
    except (AttributeError, RuntimeError, TypeError, ValueError):
        pass
    return None


def _checkpoint_from_catalog_record(
    record,
    *,
    map_id: tuple[int, int] | None,
    trainer_affordance_id: str,
) -> TrainerCheckpoint | None:
    """Convert one parsed ROM party record into the shared checkpoint type."""

    if map_id is None:
        return None
    try:
        from modules.context import context
        from modules.pokemon import get_move_by_index, get_species_by_index

        party_members = []
        for member in record.party:
            species = get_species_by_index(member.species_index)
            moves = tuple(get_move_by_index(move).name for move in member.moves if move)
            if not moves:
                learned = sorted(
                    (
                        entry
                        for entry in getattr(species.learnset, "level_up", ())
                        if int(getattr(entry, "level", 0)) <= member.level
                    ),
                    key=lambda entry: int(getattr(entry, "level", 0)),
                )
                # Gen III constructs a default trainer moveset from the
                # latest four level-up moves available at the party level.
                moves = tuple(entry.move.name for entry in learned[-4:])
            party_members.append(TrainerPokemon(species.name, member.level, moves))
        party = tuple(party_members)
        if not party:
            return None
        trainer_id = int(record.trainer_id)
        return TrainerCheckpoint(
            checkpoint_id=f"rom:{getattr(getattr(context, 'rom', None), 'id', 'active')}:{trainer_id}",
            map_id=map_id,
            trainer_affordance_id=trainer_affordance_id,
            display_name=record.name or f"Trainer {trainer_id}",
            party=party,
            source="active_rom",
            rom_trainer_id=trainer_id,
            trainer_class=record.trainer_class,
            ai_flags=record.ai_flags,
            gym_trainer="gym" in record.name.casefold(),
        )
    except (AttributeError, IndexError, KeyError, RuntimeError, TypeError, ValueError):
        return None


def _trainer_checkpoint_for_live_battle() -> TrainerCheckpoint | None:
    """Read the active ROM trainer ID and resolve its complete party."""

    try:
        from modules.memory import read_symbol, unpack_uint16
        from modules.trainer_catalog import active_trainer_catalog

        catalog = active_trainer_catalog()
        if not catalog:
            return None
        trainer_id = unpack_uint16(read_symbol("gTrainerBattleOpponent_A", size=2))
        if trainer_id < 0 or trainer_id >= len(catalog):
            return None
        record = catalog[trainer_id]
        if record is None:
            return None
        return _checkpoint_from_catalog_record(
            record,
            map_id=_current_map_id(),
            trainer_affordance_id=f"rom_trainer:{trainer_id}",
        )
    except (AttributeError, IndexError, KeyError, RuntimeError, TypeError, ValueError):
        return None


def trainer_checkpoint_for_source(
    trainer_affordance_id: str | None,
    *,
    script_symbol: str | None = None,
) -> TrainerCheckpoint | None:
    """Resolve a hand-verified or ROM-backed checkpoint for one affordance.

    The stable navigation ID remains the map/local-object identity.  The ROM
    trainer ID is resolved independently from the object's battle script so a
    map can safely contain multiple trainers with different parties.
    """

    exact = trainer_checkpoint_for_id(trainer_affordance_id)
    if exact is not None and exact.party:
        return exact
    if not trainer_affordance_id:
        return None

    try:
        from modules.memory import read_symbol
        from modules.trainer_catalog import (
            active_trainer_catalog,
            resolve_trainerbattle_id,
            resolve_trainerbattle_id_from_script,
        )

        gender, starter, context_symbols = _source_resolution_facts(exact)
        from modules.context import context

        rom_id = getattr(getattr(context, "rom", None), "id", None)
        catalog = active_trainer_catalog()
        symbols = list(context_symbols)
        if script_symbol:
            symbols.append(script_symbol)
        if exact is not None:
            if exact.script_symbol:
                symbols.append(exact.script_symbol)
            symbols.extend(exact.alternate_script_symbols)
        candidate_symbols = tuple(dict.fromkeys(symbols))
        cache_key = (
            rom_id,
            trainer_affordance_id,
            script_symbol,
            gender,
            starter,
            candidate_symbols,
            len(catalog),
        )
        if cache_key in _source_checkpoint_cache:
            return _source_checkpoint_cache[cache_key]

        def cache(value):
            if len(_source_checkpoint_cache) >= 1024:
                _source_checkpoint_cache.pop(next(iter(_source_checkpoint_cache)))
            _source_checkpoint_cache[cache_key] = value
            return value

        _source_resolution_probe(
            trainer_affordance_id,
            script_symbol,
            exact,
            catalog_size=len(catalog),
            gender=gender,
            starter=starter,
            symbols=candidate_symbols,
        )
        if not catalog:
            diagnostic_print(
                lambda: (
                    "TRAINER_CHECKPOINT_RESOLUTION: "
                    f"affordance={trainer_affordance_id!r} "
                    "result='empty_active_catalog'"
                ),
                trace=True,
                prefix="TRAINER_CHECKPOINT_RESOLUTION",
            )
            return cache(None)
        attempts = []
        for candidate_symbol in candidate_symbols:
            # Ordinary map objects commonly point at a short wrapper rather
            # than directly at the trainerbattle opcode. Keep this bounded:
            # it is enough for lock/face/call/setup prefixes and avoids
            # scanning text or unrelated script data.
            script_bytes = read_symbol(candidate_symbol, size=_TRAINER_SCRIPT_SCAN_BYTES)
            trainer_id = resolve_trainerbattle_id(
                script_bytes,
                catalog_size=len(catalog),
            )
            if trainer_id is None:
                trainer_id = resolve_trainerbattle_id_from_script(
                    script_bytes,
                    catalog_size=len(catalog),
                )
            attempts.append((candidate_symbol, script_bytes.hex(), trainer_id))
            if trainer_id is None:
                continue
            record = catalog[trainer_id]
            if record is None:
                continue
            checkpoint = _checkpoint_from_catalog_record(
                record,
                map_id=_map_id_from_affordance(trainer_affordance_id)
                or getattr(exact, "map_id", None),
                trainer_affordance_id=trainer_affordance_id,
            )
            if checkpoint is not None:
                if exact is not None:
                    checkpoint = replace(
                        checkpoint,
                        checkpoint_id=exact.checkpoint_id,
                        display_name=exact.display_name,
                        mandatory=exact.mandatory,
                        clearance_fact=exact.clearance_fact,
                        script_symbol=exact.script_symbol,
                        alternate_script_symbols=exact.alternate_script_symbols,
                        battle_script_symbols_by_context=exact.battle_script_symbols_by_context,
                        checkpoint_kind=exact.checkpoint_kind,
                    )
                diagnostic_print(
                    lambda: (
                        "TRAINER_CHECKPOINT_RESOLUTION: "
                        f"phase='success' affordance={trainer_affordance_id!r} "
                        f"script_symbol={candidate_symbol!r} trainer_id={trainer_id!r} "
                        f"checkpoint={checkpoint.checkpoint_id!r} "
                        f"party={[(member.species, member.level, member.moves) for member in checkpoint.party]!r}"
                    ),
                    trace=True,
                    prefix="TRAINER_CHECKPOINT_RESOLUTION",
                )
            return cache(checkpoint)
        diagnostic_print(
            lambda: (
                "TRAINER_CHECKPOINT_RESOLUTION: "
                f"affordance={trainer_affordance_id!r} "
                f"catalog_size={len(catalog)!r} attempts={attempts!r}"
            ),
            trace=True,
            prefix="TRAINER_CHECKPOINT_RESOLUTION",
        )
        return cache(None)
    except (AttributeError, IndexError, KeyError, RuntimeError, TypeError, ValueError) as error:
        diagnostic_print(
            lambda: (
                "TRAINER_CHECKPOINT_RESOLUTION: "
                f"affordance={trainer_affordance_id!r} result='exception' "
                f"exception_type={type(error).__name__!r} exception={error!r}"
            ),
            trace=True,
            prefix="TRAINER_CHECKPOINT_RESOLUTION",
        )
        if "cache_key" in locals():
            _source_checkpoint_cache[cache_key] = None
        return None


def trainer_checkpoints_for_overworld(overworld) -> tuple[TrainerCheckpoint, ...]:
    """Resolve every undefeated trainer currently represented by perception.

    Campaign readiness used to resolve only the trainer selected as the next
    checkpoint. That left ordinary trainers invisible in the campaign
    snapshot even when the route planner had already observed their object or
    static hazard. Object events are preferred, static trainer triggers fill
    camera/object gaps, and unresolved sources remain absent rather than
    acquiring an invented party.
    """

    if overworld is None:
        return ()
    sources: dict[str, str | None] = {}
    for obj in getattr(overworld, "objects", ()):
        trainer_id = getattr(obj, "trainer_id", None)
        # A few runtime object readers do not expose trainer_type even though
        # the object owns a script containing a trainerbattle command. Keep
        # the map/local identity stable and let the bounded ROM resolver
        # decide whether the script is actually a trainer. Source resolution
        # is cached, so this fallback does not become a per-frame ROM scan.
        if not trainer_id:
            script = getattr(obj, "script", None)
            location = getattr(obj, "location", None)
            local_id = getattr(obj, "local_id", None)
            map_id = location[0] if isinstance(location, tuple) and len(location) == 2 else None
            if script and isinstance(map_id, tuple) and isinstance(local_id, int):
                trainer_id = f"trainer:{map_id}:{local_id}"
        if not trainer_id or getattr(obj, "trainer_defeated", None) is True:
            continue
        sources.setdefault(trainer_id, getattr(obj, "script", None))
    for trigger in getattr(overworld, "triggers", ()):
        trainer_id = getattr(trigger, "affordance_id", None)
        if not trainer_id:
            continue
        if getattr(trigger, "trainer_defeated", None) is True:
            continue
        if getattr(trigger, "condition_active", True) is False:
            continue
        kind = getattr(trigger, "hazard_kind", None) or getattr(trigger, "kind", None)
        if kind not in {"trainer", "trainer_hazard"}:
            continue
        script = getattr(trigger, "script_symbol", None)
        if not script:
            trigger_id = getattr(trigger, "trigger_id", "")
            if isinstance(trigger_id, str) and trigger_id.startswith("static_object:"):
                parts = trigger_id.split(":", 2)
                script = parts[2] if len(parts) == 3 else None
        sources.setdefault(trainer_id, script)

    facts = []
    for trainer_id, script in sorted(sources.items(), key=lambda item: item[0]):
        checkpoint = trainer_checkpoint_for_source(trainer_id, script_symbol=script)
        if checkpoint is not None:
            facts.append(checkpoint)
    return tuple(facts)


def trainer_checkpoints_for_map(map_id) -> tuple[TrainerCheckpoint, ...]:
    """Resolve undefeated ROM trainer facts from a map's static templates.

    Runtime object perception is camera-scoped. Campaign readiness may already
    have a route into a later map while that map's object events are not yet
    spawned, so future trainer facts must come from the ROM object table. The
    route layer remains responsible for deciding whether a sight line is
    actually exposed; this function only supplies the roster facts.
    """

    normalized = getattr(map_id, "value", map_id)
    if not isinstance(normalized, tuple) or len(normalized) != 2:
        return ()
    try:
        from modules.map import get_map_metadata

        metadata = get_map_metadata(normalized)
    except (AttributeError, KeyError, RuntimeError, TypeError, ValueError, IndexError):
        return ()
    facts = []
    for template in getattr(metadata, "objects", ()):
        try:
            if getattr(template, "kind", "normal") != "normal":
                continue
            if getattr(template, "trainer_type", None) in (None, "None", "???"):
                continue
            try:
                defeated = template.is_trainer_defeated
            except (AttributeError, KeyError, RuntimeError, TypeError, ValueError, IndexError):
                defeated = None
            if defeated is True:
                continue
            trainer_id = f"trainer:{normalized}:{template.local_id}"
            checkpoint = trainer_checkpoint_for_source(
                trainer_id,
                script_symbol=getattr(template, "script_symbol", None),
            )
            if checkpoint is not None:
                facts.append(checkpoint)
        except (AttributeError, KeyError, RuntimeError, TypeError, ValueError, IndexError):
            continue
    # Script-owned story battles do not need to masquerade as ordinary map
    # trainers. Their semantic bindings still provide static object identity,
    # and the active ROM catalog supplies the concrete party.
    try:
        from modules.trigger_bindings import (
            TRIGGER_BINDINGS,
            read_binding_guard_values,
            resolve_trigger_binding,
        )

        guard_values = read_binding_guard_values()

        for binding in TRIGGER_BINDINGS:
            if (
                binding.map_id != normalized
                or binding.hazard_kind != "trainer"
            ):
                continue
            resolution = resolve_trigger_binding(
                binding,
                (),
                getattr(metadata, "map_size", (0, 0)),
                static_objects=getattr(metadata, "objects", ()),
                guard_values=guard_values,
            )
            if not resolution.static_available:
                continue
            trainer_id = binding.affordance_id or binding.script_symbol
            checkpoint = trainer_checkpoint_for_source(
                trainer_id,
                script_symbol=binding.script_symbol,
            )
            if checkpoint is not None:
                facts.append(checkpoint)
    except (AttributeError, KeyError, ImportError, RuntimeError, TypeError, ValueError, IndexError):
        pass
    return tuple(
        {
            trainer_checkpoint_identity(fact): fact
            for fact in facts
        }.values()
    )


def known_moves_for_battle(battle_state=None, opponent=None) -> tuple[str, ...]:
    """Return the complete known response envelope for a recognized trainer."""

    checkpoint = trainer_checkpoint_for_battle(battle_state, opponent)
    if checkpoint is None or opponent is None:
        return ()
    species = getattr(getattr(opponent, "species", None), "name", None)
    level = getattr(opponent, "level", None)
    party_index = getattr(opponent, "party_index", None)
    profile = (
        checkpoint.party[party_index]
        if isinstance(party_index, int)
        and 0 <= party_index < len(checkpoint.party)
        and checkpoint.party[party_index].species == species
        and checkpoint.party[party_index].level == level
        else next(
            (member for member in checkpoint.party if member.species == species and member.level == level),
            None,
        )
    )
    return profile.moves if profile is not None else ()
