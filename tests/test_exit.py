import pytest

from backend.simulation.exit import Exit

def test_exit_creation():
    # check that an exit stores its id and the coordinates of all its cells
    building_exit = Exit(
        id="north_exit",
        cells=((2, 0), (3, 0))
    )

    assert building_exit.id == "north_exit"
    assert building_exit.cells == ((2, 0), (3, 0))


def test_exit_rejects_invalid_definition():
    # reject an empty exit id, an exit with no cells, and repeated cell coordinates
    with pytest.raises(ValueError):
        Exit(
            id="",
            cells=((0, 0),)
        )

    with pytest.raises(ValueError):
        Exit(
            id="north_exit",
            cells=()
        )

    with pytest.raises(ValueError):
        Exit(
            id="north_exit",
            cells=((0, 0), (0, 0))
        )