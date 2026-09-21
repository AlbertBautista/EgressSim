import pytest

from backend.simulation.cell import CellType
from backend.simulation.grid import Grid

def test_grid_creation():
    # check that a new grid has the requested dimensions and every cell starts empty
    grid = Grid(5, 4)

    assert grid.width == 5
    assert grid.height == 4

    for y in range(grid.height):
        for x in range(grid.width):
            assert grid.get_cell(x, y) == CellType.EMPTY


def test_grid_rejects_invalid_dimensions():
    # reject zero or negative values for either grid dimension
    with pytest.raises(ValueError):
        Grid(0, 5)

    with pytest.raises(ValueError):
        Grid(-1, 5)

    with pytest.raises(ValueError):
        Grid(5, 0)

    with pytest.raises(ValueError):
        Grid(5, -1)

def test_set_and_get_cell():
    # check that a cell written at an x, y coordinate can be read back at that coordinate
    grid = Grid(3, 3)

    grid.set_cell(1, 2, CellType.WALL)

    assert grid.get_cell(1, 2) == CellType.WALL

def test_grid_rejects_out_of_bounds_coordinates():
    # reject reads beyond all four grid edges and writes at the width or height boundary
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