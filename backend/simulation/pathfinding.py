# uses A* to find shortest grid paths to any goal, using a supplied function to check which cells can be entered
from collections.abc import Callable
from heapq import heappop, heappush
from math import inf

from .coordinate import Coordinate


TraversabilityCheck = Callable[[int, int], bool]


def find_path(
    start: Coordinate,
    goals: frozenset[Coordinate],
    is_traversable: TraversabilityCheck,
) -> tuple[Coordinate, ...] | None:
    # return a shortest path including both endpoints, or None if no goal can be reached

    # assumes start is in bounds and traversable; the caller guarantees this
    if not goals:
        return None

    if start in goals:
        return (start,)

    # prioritize cells by estimated total cost, then cost so far, then coordinates for ties
    frontier: list[tuple[int, int, Coordinate]] = []

    start_priority = _heuristic(start, goals)
    heappush(frontier, (start_priority, 0, start))

    came_from: dict[Coordinate, Coordinate] = {}
    cost_so_far: dict[Coordinate, int] = {
        start: 0
    }

    while frontier:
        _, current_cost, current = heappop(frontier)

        # skip queued entries superseded by a shorter route to the same cell
        if current_cost != cost_so_far[current]:
            continue

        if current in goals:
            return _reconstruct_path(came_from, current)

        for neighbor in _get_neighbors(current):
            # let the caller decide which cells are in bounds and free of obstacles
            if not is_traversable(*neighbor):
                continue

            # every move costs one step; keep only routes that improve the known cost
            new_cost = current_cost + 1

            if new_cost >= cost_so_far.get(neighbor, inf):
                continue

            cost_so_far[neighbor] = new_cost
            came_from[neighbor] = current

            # A* combines steps already taken with an estimate of the steps remaining
            priority = new_cost + _heuristic(
                neighbor,
                goals,
            )

            heappush(
                frontier,
                (priority, new_cost, neighbor),
            )

    return None


def _heuristic(
    position: Coordinate,
    goals: frozenset[Coordinate],
) -> int:
    # use the smallest Manhattan distance to any goal, ignoring obstacles

    x, y = position

    return min(
        abs(x - goal_x) + abs(y - goal_y)
        for goal_x, goal_y in goals
    )


def _get_neighbors(
    position: Coordinate,
) -> tuple[Coordinate, ...]:
    # return horizontal and vertical neighbors; the caller checks whether they are usable

    x, y = position

    return (
        (x + 1, y),
        (x - 1, y),
        (x, y + 1),
        (x, y - 1),
    )


def _reconstruct_path(
    came_from: dict[Coordinate, Coordinate],
    goal: Coordinate,
) -> tuple[Coordinate, ...]:
    # follow predecessor links from goal to start, then reverse to return the path in order

    path = [goal]
    current = goal

    while current in came_from:
        current = came_from[current]
        path.append(current)

    path.reverse()

    return tuple(path)
