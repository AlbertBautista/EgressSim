from backend.simulation.building import Building
from backend.simulation.exit import Exit
from backend.simulation.pathfinding import find_path
from backend.simulation.scenario import Scenario


def assert_valid_path(path, is_traversable):
    # check that all path cells are usable and each move is one horizontal or vertical step
    assert path is not None

    for position in path:
        assert is_traversable(*position)

    for current, next_position in zip(path, path[1:]):
        x1, y1 = current
        x2, y2 = next_position

        assert abs(x1 - x2) + abs(y1 - y2) == 1


def test_find_path_in_open_grid():
    # check that an open grid yields a shortest four-step path containing both endpoints
    def is_traversable(x, y):
        return 0 <= x < 5 and 0 <= y < 5

    path = find_path(
        start=(0, 0),
        goals=frozenset({(2, 2)}),
        is_traversable=is_traversable,
    )

    assert path is not None
    assert path[0] == (0, 0)
    assert path[-1] == (2, 2)
    assert len(path) == 5

    assert_valid_path(path, is_traversable)


def test_find_path_returns_none_when_goal_is_unreachable():
    # check that a barrier spanning the grid makes the goal unreachable and returns None
    blocked_cells = {
        (2, 0),
        (2, 1),
        (2, 2),
        (2, 3),
        (2, 4),
    }

    def is_traversable(x, y):
        return (
            0 <= x < 5
            and 0 <= y < 5
            and (x, y) not in blocked_cells
        )

    path = find_path(
        start=(0, 2),
        goals=frozenset({(4, 2)}),
        is_traversable=is_traversable,
    )

    assert path is None


def test_find_path_returns_none_without_goals():
    # check that an empty set of destinations returns None
    def is_traversable(x, y):
        return 0 <= x < 5 and 0 <= y < 5

    path = find_path(
        start=(0, 0),
        goals=frozenset(),
        is_traversable=is_traversable,
    )

    assert path is None


def test_find_path_when_start_is_goal():
    # check that starting at a goal returns only the starting cell, with no movement needed
    def is_traversable(x, y):
        return 0 <= x < 5 and 0 <= y < 5

    path = find_path(
        start=(2, 2),
        goals=frozenset({(2, 2)}),
        is_traversable=is_traversable,
    )

    assert path == ((2, 2),)


def test_pathfinding_respects_walls_and_hazards():
    # check that scenario rules guide a shortest path around both walls and hazards to an exit
    building = Building(7, 5)

    building.place_wall(2, 2)

    building.add_exit(
        Exit(
            id="exit_1",
            cells=((6, 2),)
        )
    )

    scenario = Scenario(
        building=building,
        agent_specs=(),
        hazard_cells=frozenset({(3, 2)}),
    )

    path = find_path(
        start=(0, 2),
        goals=frozenset({(6, 2)}),
        is_traversable=scenario.is_traversable,
    )

    assert path is not None
    assert path[0] == (0, 2)
    assert (2, 2) not in path
    assert (3, 2) not in path
    assert path[-1] == (6, 2)
    assert len(path) == 9  # eight moves around the obstacles

    assert_valid_path(path, scenario.is_traversable)


def test_pathfinding_can_use_available_cell_of_multicell_exit():
    # check that a shortest path reaches an available exit cell when another is hazardous
    building = Building(7, 5)

    building.add_exit(
        Exit(
            id="exit_1",
            cells=(
                (6, 1),
                (6, 2),
                (6, 3),
            )
        )
    )

    scenario = Scenario(
        building=building,
        agent_specs=(),
        hazard_cells=frozenset({(6, 2)}),
    )

    building_exit = building.get_exit("exit_1")

    path = find_path(
        start=(0, 2),
        goals=frozenset(building_exit.cells),
        is_traversable=scenario.is_traversable,
    )

    assert path is not None
    assert path[0] == (0, 2)
    assert path[-1] in {
        (6, 1),
        (6, 3),
    }

    assert (6, 2) not in path
    assert len(path) == 8  # seven moves to either available exit cell

    assert_valid_path(path, scenario.is_traversable)
