"""Semantic goal bindings to authoritative runtime event identities."""

from dataclasses import dataclass
from typing import Mapping

from modules.map_data import MapRSE
from modules.memory import get_event_flag_by_number, get_event_var
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
    alternate_local_ids: tuple[int, ...] = ()
    allow_scriptless_local_match: bool = False
    # A scripted scene can own a campaign trainer checkpoint even when the
    # ROM object is not represented as a normal trainer object.  Keep the
    # stable navigation identity separate from the script symbol so campaign
    # goals can still activate the scene by its original semantic identity.
    affordance_id: str | None = None
    hazard_kind: str | None = None
    # Some script-owned objects exist in the map template before the ROM's
    # story state makes them actionable.  Keep the guard on the semantic
    # binding so static perception cannot promote an object from the wrong
    # scene.  Callers supply the save-backed values at observation time.
    guard_variable: str | None = None
    guard_operator: str = "eq"
    guard_value: int | None = None

    @property
    def script_symbols(self) -> tuple[str, ...]:
        return (self.script_symbol, *self.alternate_script_symbols)

    @property
    def local_ids(self) -> tuple[int, ...]:
        return () if self.local_id is None else (self.local_id, *self.alternate_local_ids)

    def matches_object(self, local_id: int | None, script_symbol: str | None) -> bool:
        local_match = self.local_id is None or local_id in self.local_ids
        script_match = script_symbol in self.script_symbols
        scriptless_local_match = (
            self.allow_scriptless_local_match
            and self.local_id is not None
            and local_id in self.local_ids
            and not script_symbol
        )
        return local_match and (script_match or scriptless_local_match)


@dataclass(frozen=True)
class BindingResolution:
    binding: TriggerBinding
    static_match: bool = False
    static_location: tuple[tuple[int, int], tuple[int, int]] | None = None
    # A script identity can have several ROM object templates.  Keep every
    # visible template so a semantic hazard remains usable before the runtime
    # object is camera-scoped.  ``static_location`` remains the single-template
    # compatibility view; callers that need complete geometry must use this
    # collection.
    static_locations: tuple[tuple[tuple[int, int], tuple[int, int]], ...] = ()
    static_available: bool = False
    # ``None`` means that no static template was selected or that its
    # visibility flag could not be read.  Keep that distinct from ``False``
    # (the ROM positively reports the object hidden) so a transient save/map
    # read cannot publish an open interaction affordance.
    static_availability: bool | None = None
    static_ambiguous: bool = False
    runtime_match: bool = False
    object_ids: tuple[int, ...] = ()
    runtime_locations: tuple[tuple[int, int], ...] = ()
    interaction_positions: tuple[tuple[int, int], ...] = ()
    guard_current_value: int | None = None
    guard_active: bool | None = None


# The map, local ID, and script symbol are ROM-backed identities.  Coordinates
# come from the map template so they remain available before runtime spawn.
TRIGGER_BINDINGS: tuple[TriggerBinding, ...] = (
    TriggerBinding(
        trigger_id="introductory_rival",
        map_id=MapRSE.ROUTE103.value,
        script_symbol="Route103_EventScript_Rival",
        local_id=2,
        affordance_id=f"trainer:{MapRSE.ROUTE103.value}:2",
        hazard_kind="trainer",
        # pret/pokeemerald 5eff78649e7170a877b961ef0b3da13b81a16038:
        # Birch's lab sets this to 3 after the starter handoff, and the
        # Route 103 rival end script sets it to 4.  Equality prevents a
        # stale/static object template from being treated as the active rival
        # scene before the starter handoff or after the battle boundary.
        guard_variable="BIRCH_LAB_STATE",
        guard_operator="eq",
        guard_value=3,
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
        # retained as the source-backed fallback because this ROM's live
        # templates use an empty script pointer for the two scene objects.
        script_symbol="PetalburgWoods_EventScript_DevonResearcherLeft",
        alternate_script_symbols=(
            "PetalburgWoods_EventScript_DevonResearcherRight",
            # The initial script is also used as the object identity by some
            # ROM object/template revisions before it dispatches to the
            # approach-side script.
            "PetalburgWoods_EventScript_DevonResearcherIntro",
        ),
        local_id=4,
        alternate_local_ids=(3,),
        allow_scriptless_local_match=True,
        affordance_id="devon_goods_researcher",
        hazard_kind="trainer",
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
        affordance_id="rusturf_tunnel_goods",
        hazard_kind="trainer",
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


def read_binding_guard_values(
    bindings: tuple[TriggerBinding, ...] = TRIGGER_BINDINGS,
) -> dict[str, int | None]:
    """Read save-backed availability guards for semantic bindings.

    A failed read is retained as ``None``.  Callers pass the resulting mapping
    to :func:`resolve_trigger_binding`, which then withholds guarded geometry
    until the ROM state is authoritative.
    """

    values: dict[str, int | None] = {}
    for binding in bindings:
        variable = binding.guard_variable
        if variable is None or variable in values:
            continue
        try:
            values[variable] = get_event_var(variable)
        except (AttributeError, KeyError, RuntimeError, TypeError, ValueError, IndexError):
            values[variable] = None
    return values


def resolve_trigger_binding(
    binding: TriggerBinding,
    objects,
    map_size: tuple[int, int],
    static_objects=None,
    guard_values: Mapping[str, int | None] | None = None,
) -> BindingResolution:
    total_start = now()
    count("trigger_static_resolution_calls")
    if static_objects is None:
        static_objects = ()

    guard_current_value = None
    guard_active: bool | None = True
    if binding.guard_variable is not None:
        # ``None`` means the caller is using this low-level resolver for a
        # pure geometry calculation.  Live perception and static trainer
        # catalog callers pass a mapping; those callers must fail closed when
        # the authoritative variable is unavailable.
        if guard_values is not None:
            guard_current_value = guard_values.get(binding.guard_variable)
            if guard_current_value is None or binding.guard_value is None:
                guard_active = None
            elif binding.guard_operator == "eq":
                guard_active = guard_current_value == binding.guard_value
            elif binding.guard_operator == "ne":
                guard_active = guard_current_value != binding.guard_value
            elif binding.guard_operator == "ge":
                guard_active = guard_current_value >= binding.guard_value
            elif binding.guard_operator == "gt":
                guard_active = guard_current_value > binding.guard_value
            elif binding.guard_operator == "le":
                guard_active = guard_current_value <= binding.guard_value
            elif binding.guard_operator == "lt":
                guard_active = guard_current_value < binding.guard_value
            else:
                guard_active = None

    static_start = now()
    static_matches = tuple(
        object_template
        for object_template in static_objects
        if binding.matches_object(object_template.local_id, object_template.script_symbol)
    )
    static_match = bool(static_matches)
    timing("trigger_static_template_matching", static_start)
    count("static_template_resolutions")
    hide_flag_start = now()
    static_template_availability: list[tuple[object, bool | None]] = []
    for static_template in static_matches:
        hide_flag_read_start = now()
        try:
            static_flag_id = static_template.flag_id
        except (AttributeError, KeyError, RuntimeError, TypeError, ValueError, IndexError):
            static_flag_id = None
        if static_flag_id == 0:
            template_availability = True
        elif static_flag_id is None:
            template_availability = None
        else:
            try:
                observed_hide_flag = get_event_flag_by_number(static_flag_id)
            except (AttributeError, KeyError, RuntimeError, TypeError, ValueError, IndexError):
                observed_hide_flag = None
            if observed_hide_flag is True:
                template_availability = False
            elif observed_hide_flag is False:
                template_availability = True
            else:
                # Any other result is unavailable/unknown.  It must not
                # become an implicit ``False`` through bool(None).
                template_availability = None
            timing("trigger_hide_flag_emulator_read", hide_flag_read_start)
        static_template_availability.append((static_template, template_availability))

    availability_values = tuple(value for _, value in static_template_availability)
    if availability_values and all(value is not None for value in availability_values):
        static_availability = any(value is True for value in availability_values)
    elif availability_values:
        static_availability = None
    else:
        static_availability = None
    static_available = static_availability is True
    timing("trigger_hide_flag_evaluation", hide_flag_start)
    count("hide_flag_evaluations")
    static_locations = (
        tuple(
            (binding.map_id, static_template.local_coordinates)
            for static_template, template_availability in static_template_availability
            if template_availability is True
        )
        if static_available
        else ()
    )
    static_location = static_locations[0] if len(static_locations) == 1 else None

    if binding.guard_variable is not None and guard_values is not None and guard_active is not True:
        return BindingResolution(
            binding=binding,
            static_match=static_match,
            static_available=False,
            static_availability=static_availability,
            static_ambiguous=len(static_matches) > 1,
            guard_current_value=guard_current_value,
            guard_active=guard_active,
        )
    runtime_start = now()
    matches = (
        ()
        if static_match and static_availability is not True
        else tuple(
            object_observation
            for object_observation in objects
            if object_observation.location[0] == binding.map_id
            and binding.matches_object(object_observation.local_id, object_observation.script)
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
    if not interaction_positions and static_available:
        interaction_positions = tuple(
            sorted(
                {
                    (candidate_x, candidate_y)
                    for _, (object_x, object_y) in static_locations
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
        static_locations=static_locations,
        static_available=static_available,
        static_availability=static_availability,
        static_ambiguous=len(static_matches) > 1,
        runtime_match=bool(matches),
        object_ids=tuple(object_observation.local_id for object_observation in matches),
        runtime_locations=tuple(object_observation.location[1] for object_observation in matches),
        interaction_positions=interaction_positions,
        guard_current_value=guard_current_value,
        guard_active=guard_active,
    )
