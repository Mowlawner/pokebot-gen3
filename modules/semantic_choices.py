"""Frame-local semantic choices and conservative consequence-based selection.

This module deliberately does not know how a menu is navigated.  A ROM adapter
may attach the consequences it can establish; policy selects an outcome only
when that evidence is sufficient.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping


class ConsequenceKnowledge(Enum):
    KNOWN = "known"
    PARTIALLY_KNOWN = "partially_known"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class ChoiceConsequence:
    option: str
    knowledge: ConsequenceKnowledge = ConsequenceKnowledge.UNKNOWN
    fact_updates: Mapping[str, object] = field(default_factory=dict)
    interaction_complete: bool | None = None
    description: str | None = None


@dataclass(frozen=True, slots=True)
class DialogueChoice:
    question_text: str | None
    options: tuple[str, ...]
    selected_option: str | None = None
    source_affordance: str | None = None
    script_identity: str | None = None
    consequences: tuple[ChoiceConsequence, ...] = ()

    def consequence_for(self, option: str) -> ChoiceConsequence | None:
        return next((item for item in self.consequences if item.option == option), None)


def choose_choice_for_outcome(
    choice: DialogueChoice,
    *,
    current_facts: Mapping[str, object],
    desired_facts: Mapping[str, object],
) -> str | None:
    """Return the option with a known, objective-advancing consequence.

    Unknown and merely partial predictions are never promoted to certainty.
    If the desired facts are already true, or no uniquely supported option is
    available, return ``None`` and let the caller re-observe or defer.
    """
    if desired_facts and all(current_facts.get(name) == value for name, value in desired_facts.items()):
        return None
    candidates: list[str] = []
    for option in choice.options:
        consequence = choice.consequence_for(option)
        if consequence is None or consequence.knowledge is not ConsequenceKnowledge.KNOWN:
            continue
        predicted = dict(current_facts)
        predicted.update(consequence.fact_updates)
        if all(predicted.get(name) == value for name, value in desired_facts.items()):
            candidates.append(option)
    return candidates[0] if len(candidates) == 1 else None


__all__ = [
    "ChoiceConsequence",
    "ConsequenceKnowledge",
    "DialogueChoice",
    "choose_choice_for_outcome",
]
