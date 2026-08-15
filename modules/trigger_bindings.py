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
        and object_template.script_symbol == binding.script_symbol
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
            and object_observation.script == binding.script_symbol
        )
    )
    timing("trigger_runtime_binding_matching", runtime_start)
    count("runtime_binding_matches")
    geometry_start = now()
    interaction_positions = tuple(
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
