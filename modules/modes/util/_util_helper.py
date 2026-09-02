from functools import wraps

from modules.context import context
from modules.console import diagnostic_print


def isolate_inputs(generator_function):
    @wraps(generator_function)
    def wrapper_function(*args, **kwargs):
        previous_inputs = context.emulator.reset_held_buttons()
        is_battle_handler = generator_function.__qualname__ in {
            "BattleListener.fight",
            "BattleListener.catch",
            "BattleListener.run_away_from_battle",
        }
        yield from generator_function(*args, **kwargs)
        if is_battle_handler:
            diagnostic_print("STARTER_FLOW: isolate_inputs fight wrapper returned", trace=True)
            # Battle completion notifies the active mode before this wrapper
            # returns. That callback deliberately releases movement input so
            # the post-battle controller starts from a neutral edge. Restoring
            # the pre-battle direction here reintroduces the stale route input
            # and can immediately walk into the trainer that just triggered
            # the battle (or into a newly moved trainer).
            context.emulator.reset_held_buttons()
        else:
            context.emulator.restore_held_buttons(previous_inputs)

    return wrapper_function
