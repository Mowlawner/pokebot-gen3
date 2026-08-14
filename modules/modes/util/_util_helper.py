from functools import wraps

from modules.context import context
from modules.console import diagnostic_print


def isolate_inputs(generator_function):
    @wraps(generator_function)
    def wrapper_function(*args, **kwargs):
        previous_inputs = context.emulator.reset_held_buttons()
        is_battle_fight = generator_function.__qualname__ == "BattleListener.fight"
        yield from generator_function(*args, **kwargs)
        if is_battle_fight:
            diagnostic_print("STARTER_FLOW: isolate_inputs fight wrapper returned", trace=True)
        context.emulator.restore_held_buttons(previous_inputs)

    return wrapper_function
