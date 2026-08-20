import sys
from modules.navigation import NavigationWorld, NavigableTile, GoalAwareNavigator, NavigationActionType
from modules.goals import ReachLocation


# Minimal setup to inspect costs
def debug_costs():
    # Construct a tiny world
    # Littleroot town map: (0,9)
    # (5,9) is where we start.
    # We want to go to (7,9) (Rival house area).

    # Coordinates:
    start = ((0, 9), (5, 9))
    goal = ((0, 9), (7, 9))

    # Define tiles
    tiles = {
        ((0, 9), (5, 9)): NavigableTile(((0, 9), (5, 9)), has_encounters=False),
        ((0, 9), (6, 9)): NavigableTile(((0, 9), (6, 9)), has_encounters=True),  # Suspect encounter tile
        ((0, 9), (7, 9)): NavigableTile(((0, 9), (7, 9)), has_encounters=False),
        ((0, 9), (5, 10)): NavigableTile(((0, 9), (5, 10)), has_encounters=False),
        ((0, 9), (6, 10)): NavigableTile(((0, 9), (6, 10)), has_encounters=False),
    }

    world = NavigationWorld(tiles=tiles, warps=(), triggers=())
    navigator = GoalAwareNavigator(world)

    # Plan
    plan = navigator.plan(start, ReachLocation(goal))
    print(f"Plan: {plan}")
    if plan:
        print(f"Metrics: {plan.metrics}")


if __name__ == "__main__":
    debug_costs()
