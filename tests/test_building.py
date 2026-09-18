import pytest

from backend.simulation.building import Building
from backend.simulation.cell import CellType
from backend.simulation.exit import Exit


def test_wall_placement_and_removal():
    building = Building(5, 4)

    building.place_wall(2, 2)

    assert building.get_cell(2, 2) == CellType.WALL

    building.remove_wall(2, 2)

    assert building.get_cell(2, 2) == CellType.EMPTY


def test_add_exit():
    building = Building(6, 5)

    bunker_exit = Exit(
        id="bunker_exit",
        cells=(
            (2, 2),
            (3, 2),
            (4, 2),
            (2, 3),
            (3, 3),
            (4, 3),
        )
    )

    building.add_exit(bunker_exit)

    assert building.get_exit("bunker_exit") == bunker_exit

    for x, y in bunker_exit.cells:
        assert building.get_exit_at(x, y) == bunker_exit
        assert building.get_cell(x, y) == CellType.EXIT


def test_remove_exit():
    building = Building(6, 4)

    building_exit = Exit(
        id="exit_1",
        cells=((2, 1), (3, 1))
    )

    building.add_exit(building_exit)
    building.remove_exit("exit_1")

    assert building.exits == ()
    assert building.get_exit_at(2, 1) is None
    assert building.get_exit_at(3, 1) is None

    assert building.get_cell(2, 1) == CellType.EMPTY
    assert building.get_cell(3, 1) == CellType.EMPTY


@pytest.mark.parametrize(
    "cells",
    [
        ((1, 1), (3, 1)),
        ((1, 1), (2, 2)),
        ((1, 1), (1, 2), (3, 2)),
        ((6, 1),),
    ],
)
def test_building_rejects_invalid_exit_placement(cells):
    building = Building(6, 4)

    building_exit = Exit(
        id="invalid_exit",
        cells=cells
    )

    with pytest.raises(ValueError):
        building.add_exit(building_exit)


def test_building_rejects_duplicate_exit_ids():
    building = Building(6, 4)

    building.add_exit(
        Exit(
            id="exit_1",
            cells=((0, 1),)
        )
    )

    with pytest.raises(ValueError):
        building.add_exit(
            Exit(
                id="exit_1",
                cells=((5, 1),)
            )
        )


def test_building_prevents_geometry_overlap():
    building = Building(6, 4)

    building.place_wall(0, 1)

    with pytest.raises(ValueError):
        building.add_exit(
            Exit(
                id="west_exit",
                cells=((0, 1),)
            )
        )

    building.add_exit(
        Exit(
            id="east_exit",
            cells=((5, 1),)
        )
    )

    with pytest.raises(ValueError):
        building.place_wall(5, 1)