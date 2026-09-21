# manages permanent building geometry, enforcing wall and exit placement rules on a grid
from .cell import CellType
from .coordinate import Coordinate
from .exit import Exit
from .grid import Grid


class Building:

    def __init__(self, width: int, height: int):
        # create an empty building with the given dimensions

        self._grid = Grid(width, height)

        self._exits: dict[str, Exit] = {}
        self._exit_by_cell: dict[Coordinate, str] = {}

    @property
    def width(self) -> int:
        return self._grid.width

    @property
    def height(self) -> int:
        return self._grid.height

    @property
    def exits(self) -> tuple[Exit, ...]:
        # return all exits registered in the building

        return tuple(self._exits.values())

    def is_in_bounds(self, x: int, y: int) -> bool:
        # return True if the coordinates are inside the building

        return self._grid.is_in_bounds(x, y)

    def get_cell(self, x: int, y: int) -> CellType:
        # return the cell type at the given coordinates

        return self._grid.get_cell(x, y)

    def place_wall(self, x: int, y: int) -> None:
        # place a wall on an empty cell

        if self._grid.get_cell(x, y) != CellType.EMPTY:
            raise ValueError("Walls can only be placed on empty cells.")

        self._grid.set_cell(x, y, CellType.WALL)

    def remove_wall(self, x: int, y: int) -> None:
        # restore a wall cell to empty, rejecting cells that do not contain a wall
        if self._grid.get_cell(x, y) != CellType.WALL:
            raise ValueError("There is no wall at these coordinates.")

        self._grid.set_cell(x, y, CellType.EMPTY)

    def add_exit(self, building_exit: Exit) -> None:
        # validate the entire exit before registering it and marking its cells on the grid
        if building_exit.id in self._exits:
            raise ValueError(
                f"An exit with id '{building_exit.id}' already exists."
            )

        for x, y in building_exit.cells:
            if not self._grid.is_in_bounds(x, y):
                raise ValueError("Exit cells must be inside the building.")

        self._validate_exit_shape(building_exit)

        for x, y in building_exit.cells:
            if self._grid.get_cell(x, y) != CellType.EMPTY:
                raise ValueError("Exit cells must be empty.")

        self._exits[building_exit.id] = building_exit

        # keep coordinate lookups and grid cell types synchronized with the exit registry
        for x, y in building_exit.cells:
            self._exit_by_cell[(x, y)] = building_exit.id
            self._grid.set_cell(x, y, CellType.EXIT)

    def remove_exit(self, exit_id: str) -> None:
        # verify every exit cell is consistent before clearing its cells and lookup entries
        building_exit = self._exits[exit_id]

        for cell in building_exit.cells:
            if self._exit_by_cell.get(cell) != exit_id:
                raise RuntimeError("Building exit data is inconsistent.")

            x, y = cell

            if self._grid.get_cell(x, y) != CellType.EXIT:
                raise RuntimeError("Building exit data is inconsistent.")

        for x, y in building_exit.cells:
            del self._exit_by_cell[(x, y)]
            self._grid.set_cell(x, y, CellType.EMPTY)

        del self._exits[exit_id]

    def get_exit(self, exit_id: str) -> Exit:
        # return an exit by its id

        return self._exits[exit_id]

    def get_exit_at(self, x: int, y: int) -> Exit | None:
        # return the exit occupying a coordinate, if one exists

        if not self._grid.is_in_bounds(x, y):
            raise IndexError("Grid coordinates are out of bounds.")

        exit_id = self._exit_by_cell.get((x, y))

        if exit_id is None:
            return None

        return self._exits[exit_id]

    def _validate_exit_shape(self, building_exit: Exit) -> None:
        # require one region connected through up, down, left, or right neighbors

        exit_cells = set(building_exit.cells)

        start = building_exit.cells[0]
        visited: set[Coordinate] = set()
        stack = [start]

        # use a depth-first search to find all exit cells reachable from the first cell
        while stack:
            x, y = stack.pop()

            if (x, y) in visited:
                continue

            visited.add((x, y))

            neighbors = (
                (x + 1, y),
                (x - 1, y),
                (x, y + 1),
                (x, y - 1),
            )

            for neighbor in neighbors:
                if neighbor in exit_cells and neighbor not in visited:
                    stack.append(neighbor)

        # any unvisited exit cells belong to a disconnected part of the exit
        if visited != exit_cells:
            raise ValueError(
                "Exit cells must form one connected region."
            )
