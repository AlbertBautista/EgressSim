import pytest

from backend.simulation.cell import CellType
from backend.simulation.grid import Grid

def test_grid_creation():
    grid = Grid(5, 4)

    assert grid.width == 5
    assert grid.height == 4

    for y in range(grid.height):
        for x in range(grid.width):
            assert grid.get_cell(x, y) == CellType.EMPTY


def test_grid_rejects_invalid_dimensions():
    with pytest.raises(ValueError):
        Grid(0, 5)

    with pytest.raises(ValueError):
        Grid(-1, 5)

    with pytest.raises(ValueError):
        Grid(5, 0)

    with pytest.raises(ValueError):
        Grid(5, -1)

def test_set_and_get_cell():
    grid = Grid(3, 3)

    grid.set_cell(1, 2, CellType.WALL)

    assert grid.get_cell(1, 2) == CellType.WALL

def test_grid_rejects_out_of_bounds_coordinates():
    grid = Grid(5, 3)

    with pytest.raises(IndexError):
        grid.get_cell(5, 1)

    with pytest.raises(IndexError):
        grid.get_cell(-1, 1)

    with pytest.raises(IndexError):
        grid.get_cell(1, 3)

    with pytest.raises(IndexError):
        grid.get_cell(1, -1)

    with pytest.raises(IndexError):
        grid.set_cell(5, 1, CellType.WALL)

    with pytest.raises(IndexError):
        grid.set_cell(1, 3, CellType.WALL)