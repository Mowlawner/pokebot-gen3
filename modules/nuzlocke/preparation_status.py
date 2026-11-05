"""Human-readable descriptions of campaign preparation navigation targets."""

from __future__ import annotations

from dataclasses import dataclass
import re

from modules.goals import EngageTrainer, NavigationGoal, ReachLocation, SemanticTarget
from modules.map_formatting import map_display_name


@dataclass(frozen=True, slots=True)
class PreparationTarget:
    """A semantic preparation target and the text shown to an operator."""

    semantic_target: SemanticTarget
    description: str


def _humanize_script(script: str | None) -> str | None:
    if not script:
        return None
    label = script.split("_EventScript_", 1)[-1]
    label = label.replace("_", " ")
    label = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", label)
    label = " ".join(label.split())
    return label or None


def _trainer_source(overworld, trainer_id: str):
    for obj in getattr(overworld, "objects", ()):
        if getattr(obj, "trainer_id", None) == trainer_id:
            return obj
    for trigger in getattr(overworld, "triggers", ()):
        if getattr(trigger, "affordance_id", None) == trainer_id:
            return trigger
    return None


def _trainer_coordinates(source):
    if source is None:
        return None
    location = getattr(source, "location", None)
    if location is not None:
        return location[1]
    for field in ("locations", "hazard_locations", "engagement_locations"):
        locations = tuple(getattr(source, field, ()) or ())
        if locations:
            return sorted(locations)[0][1]
    return None


def _trainer_script(source):
    script = getattr(source, "script", None) or getattr(source, "script_symbol", None)
    if script:
        return script
    trigger_id = getattr(source, "trigger_id", "")
    if trigger_id.startswith("static_object:"):
        return trigger_id.split(":", 2)[-1]
    return None


def _trainer_identity(source, script: str | None) -> str:
    script_label = _humanize_script(script)
    if script_label and script_label.casefold() not in {"trainer", "npc"}:
        return script_label
    trainer_type = getattr(source, "trainer_type", None)
    if trainer_type and trainer_type not in {"None", "???"}:
        return f"{trainer_type} trainer"
    return "trainer"


def describe_preparation_target(navigation_goal, overworld) -> PreparationTarget | None:
    """Describe a selected grass tile or trainer from the live observation."""

    goal = navigation_goal
    if isinstance(goal, NavigationGoal) and isinstance(goal.target, ReachLocation):
        # A trainer-safe grass route is a ReachLocation wrapped with its AVOID
        # constraints.  It describes the same training tile as a bare goal.
        goal = goal.target
    if isinstance(goal, ReachLocation):
        map_id, coordinates = goal.location
        map_name = map_display_name(map_id) or repr(map_id)
        description = f"grass on {map_name} at {coordinates!r}"
        return PreparationTarget(SemanticTarget.at(goal.location), description)

    if not isinstance(goal, NavigationGoal) or not isinstance(goal.target, EngageTrainer):
        return None

    trainer_id = goal.target.trainer_id
    source = _trainer_source(overworld, trainer_id)
    source_location = getattr(source, "location", None)
    map_id = source_location[0] if source_location is not None else getattr(overworld, "map_id", None)
    coordinates = _trainer_coordinates(source)
    script = _trainer_script(source)
    identity = _trainer_identity(source, script)
    map_name = map_display_name(map_id) or repr(map_id)
    coordinate_text = f" at {coordinates!r}" if coordinates is not None else ""
    metadata = []
    if script:
        metadata.append(f"script={script}")
    metadata.append(f"id={trainer_id}")
    description = f"{identity} on {map_name}{coordinate_text} [{', '.join(metadata)}]"
    return PreparationTarget(SemanticTarget.interaction(map_id, description), description)


__all__ = ["PreparationTarget", "describe_preparation_target", "map_display_name"]
