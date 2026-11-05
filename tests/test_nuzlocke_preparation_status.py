from types import SimpleNamespace

from modules.goals import EngageTrainer, NavigationGoal, ReachLocation
from modules.nuzlocke.preparation_status import describe_preparation_target
from modules.overworld import ObjectObservation, TriggerObservation


def test_grass_preparation_target_includes_map_name_and_tile():
    result = describe_preparation_target(
        ReachLocation(((0, 31), (12, 17))),
        SimpleNamespace(map_id=(0, 31), objects=(), triggers=()),
    )

    assert result.description == "grass on Route 116 at (12, 17)"
    assert result.semantic_target.location == ((0, 31), (12, 17))


def test_trainer_safe_grass_goal_is_described_like_a_bare_grass_goal():
    """A trainer-safe grass route wraps its ReachLocation in a NavigationGoal.

    The wild-battle policy treats a published preparation target as "training
    requested".  If the wrapped goal is not described, no target is published
    and every wild battle during preparation is fled, so no XP is ever earned.
    """

    from modules.goals import GoalConstraints, TrainerMode

    wrapped = NavigationGoal(
        ReachLocation(((0, 31), (12, 17))),
        constraints=GoalConstraints(trainer_mode=TrainerMode.AVOID),
    )
    result = describe_preparation_target(
        wrapped,
        SimpleNamespace(map_id=(0, 31), objects=(), triggers=()),
    )

    assert result is not None
    assert result.description == "grass on Route 116 at (12, 17)"
    assert result.semantic_target.location == ((0, 31), (12, 17))


def test_trainer_preparation_target_uses_script_and_stable_id():
    trainer = ObjectObservation(
        local_id=6,
        location=((0, 31), (12, 17)),
        script="Route116_EventScript_Trainer",
        trainer_type="Normal",
        trainer_defeated=False,
    )
    result = describe_preparation_target(
        NavigationGoal(EngageTrainer(trainer.trainer_id)),
        SimpleNamespace(map_id=(0, 31), objects=(trainer,), triggers=()),
    )

    assert result.description == (
        "Normal trainer on Route 116 at (12, 17) "
        "[script=Route116_EventScript_Trainer, id=trainer:(0, 31):6]"
    )
    assert result.semantic_target.interaction_id == result.description


def test_static_trainer_target_uses_trigger_script_when_object_is_not_loaded():
    trainer_id = "trainer:(0, 31):6"
    trigger = TriggerObservation(
        trigger_id="static_object:6:Route116_EventScript_Trainer",
        locations=frozenset({((0, 31), (12, 17))}),
        affordance_id=trainer_id,
        hazard_kind="trainer",
    )
    result = describe_preparation_target(
        NavigationGoal(EngageTrainer(trainer_id)),
        SimpleNamespace(map_id=(0, 31), objects=(), triggers=(trigger,)),
    )

    assert "Route116_EventScript_Trainer" in result.description
    assert "trainer:(0, 31):6" in result.description
