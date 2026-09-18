import pytest

from backend.simulation.exit import Exit

def test_exit_creation():
    building_exit = Exit(
        id="north_exit",
        cells=((2, 0), (3, 0))
    )

    assert building_exit.id == "north_exit"
    assert building_exit.cells == ((2, 0), (3, 0))


def test_exit_rejects_invalid_definition():
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