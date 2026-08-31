"""Semantic goal bindings to authoritative runtime event identities."""

from dataclasses import dataclass

from modules.map_data import MapRSE
from modules.memory import get_event_flag_by_number
from modules.profiler import count, now, timing


@dataclass(frozen=True)
class TriggerBinding:
    """Identity of a semantic trigger in the ROM/runtime domain."""

    trigger_id: str
    map_id: tuple[int, int]
    script_symbol: str
    event_type: str = "object"
    local_id: int | None = None
    alternate_script_symbols: tuple[str, ...] = ()

    @property
    def script_symbols(self) -> tuple[str, ...]:
        return (self.script_symbol, *self.alternate_script_symbols)


@dataclass(frozen=True)
class BindingResolution:
    binding: TriggerBinding
    static_match: bool = False
    static_location: tuple[tuple[int, int], tuple[int, int]] | None = None
    static_available: bool = False
    static_ambiguous: bool = False
    runtime_match: bool = False
    object_ids: tuple[int, ...] = ()
    runtime_locations: tuple[tuple[int, int], ...] = ()
    interaction_positions: tuple[tuple[int, int], ...] = ()


# The map, local ID, and script symbol are ROM-backed identities.  Coordinates
# come from the map template so they remain available before runtime spawn.
TRIGGER_BINDINGS: tuple[TriggerBinding, ...] = (
    TriggerBinding(
        trigger_id="introductory_rival",
        map_id=MapRSE.ROUTE103.value,
        script_symbol="Route103_EventScript_Rival",
        local_id=2,
    ),
    TriggerBinding(
        trigger_id="early_pokeballs",
        map_id=MapRSE.LITTLEROOT_TOWN_PROFESSOR_BIRCHS_LAB.value,
        # Birch is the live object that owns this interaction.  The gender
        # scripts are variants of the interaction, not the object's identity.
        script_symbol="LittlerootTown_ProfessorBirchsLab_EventScript_Birch",
        local_id=2,
        alternate_script_symbols=(
            "LittlerootTown_ProfessorBirchsLab_EventScript_BrendanGivePokeBalls",
            "LittlerootTown_ProfessorBirchsLab_EventScript_MayGivePokeBalls",
        ),
    ),
    TriggerBinding(
        trigger_id="devon_goods_researcher",
        map_id=MapRSE.PETALBURG_WOODS.value,
        # The decompilation has left/right script variants for the same
        # researcher scene.  Script identity is authoritative; local ID is
        # intentionally omitted because the scene's object template can vary
        # with the player's approach side.
        script_symbol="PetalburgWoods_EventScript_DevonResearcherLeft",
        alternate_script_symbols=("PetalburgWoods_EventScript_DevonResearcherRight",),
    ),
    TriggerBinding(
        trigger_id="petalburg_norman",
        map_id=MapRSE.PETALBURG_CITY_GYM.value,
        script_symbol="PetalburgCity_Gym_EventScript_Norman",
        local_id=1,
    ),
    TriggerBinding(
        trigger_id="rusturf_tunnel_goods",
        map_id=MapRSE.RUSTURF_TUNNEL.value,
        script_symbol="RusturfTunnel_EventScript_Grunt",
    ),
    TriggerBinding(
        trigger_id="rustboro_return_devon_goods",
        map_id=MapRSE.RUSTBORO_CITY.value,
        script_symbol="RustboroCity_EventScript_DevonEmployee1",
    ),
    TriggerBinding(
        trigger_id="roxanne",
        map_id=MapRSE.RUSTBORO_CITY_GYM.value,
        script_symbol="RustboroCity_Gym_EventScript_Roxanne",
    ),
)


def get_trigger_binding(trigger_id: str) -> TriggerBinding | None:
    return next((binding for binding in TRIGGER_BINDINGS if binding.trigger_id == trigger_id), None)


def resolve_trigger_binding(
    binding: TriggerBinding,
    objects,
    map_size: tuple[int, int],
    static_objects=None,
) -> BindingResolution:
    total_start = now()
    count("trigger_static_resolution_calls")
    if static_objects is None:
        static_objects = ()

    static_start = now()
    static_matches = tuple(
        object_template
        for object_template in static_objects
        if (binding.local_id is None or object_template.local_id == binding.local_id)
        and object_template.script_symbol in binding.script_symbols
    )
    static_match = bool(static_matches)
    timing("trigger_static_template_matching", static_start)
    count("static_template_resolutions")
    static_template = static_matches[0] if len(static_matches) == 1 else None
    hide_flag_start = now()
    hide_flag_read_start = now()
    hide_flag_set = False
    if static_template is not None and static_template.flag_id != 0:
        hide_flag_set = get_event_flag_by_number(static_template.flag_id)
        timing("trigger_hide_flag_emulator_read", hide_flag_read_start)
    static_available = bool(static_template is not None and not hide_flag_set)
    timing("trigger_hide_flag_evaluation", hide_flag_start)
    count("hide_flag_evaluations")
    static_location = (binding.map_id, static_template.local_coordinates) if static_available else None
    runtime_start = now()
    matches = (
        ()
        if static_match and not static_available
        else tuple(
            object_observation
            for object_observation in objects
            if object_observation.location[0] == binding.map_id
            and (binding.local_id is None or object_observation.local_id == binding.local_id)
            and object_observation.script in binding.script_symbols
        )
    )
    timing("trigger_runtime_binding_matching", runtime_start)
    count("runtime_binding_matches")
    geometry_start = now()
    runtime_interaction_positions = tuple(
        sorted(
            {
                (candidate_x, candidate_y)
                for object_observation in matches
                for object_x, object_y in (object_observation.location[1],)
                for candidate_x, candidate_y in (
                    (object_x, object_y - 1),
                    (object_x + 1, object_y),
                    (object_x, object_y + 1),
                    (object_x - 1, object_y),
                )
                if 0 <= candidate_x < map_size[0] and 0 <= candidate_y < map_size[1]
            }
        )
    )
    # Runtime ObjectEvents are normally the authoritative position, but an
    # object can be absent briefly while a map/script finishes spawning it.
    # Keep the static template useful in that interval: the template's
    # coordinate is stable and the interaction geometry is the same adjacent
    # four-tile geometry used for a live object.  Do not use this fallback for
    # hidden or ambiguous templates, and always prefer the live position when
    # one is available.
    interaction_positions = runtime_interaction_positions
    if not interaction_positions and static_available and static_template is not None:
        object_x, object_y = static_template.local_coordinates
        interaction_positions = tuple(
            sorted(
                {
                    (candidate_x, candidate_y)
                    for candidate_x, candidate_y in (
                        (object_x, object_y - 1),
                        (object_x + 1, object_y),
                        (object_x, object_y + 1),
                        (object_x - 1, object_y),
                    )
                    if 0 <= candidate_x < map_size[0] and 0 <= candidate_y < map_size[1]
                }
            )
        )
    timing("trigger_geometry_matching", geometry_start)
    count("trigger_geometry_matches")
    timing("trigger_static_resolution", total_start)
    return BindingResolution(
        binding=binding,
        static_match=static_match,
        static_location=static_location,
        static_available=static_available,
        static_ambiguous=len(static_matches) > 1,
        runtime_match=bool(matches),
        object_ids=tuple(object_observation.local_id for object_observation in matches),
        runtime_locations=tuple(object_observation.location[1] for object_observation in matches),
        interaction_positions=interaction_positions,
    )
