# stores cell types in a 2D grid and checks coordinate bounds for reads and writes
from .cell import CellType

class Grid:
    def __init__(self, width: int, height: int):
        # create an empty grid with positive dimensions, storing rows before columns
        if width <= 0 or height <= 0:
            raise ValueError("Grid dimensions must be positive.")

        self.width = width
        self.height = height

        self.cells = [
            [CellType.EMPTY for _ in range(width)]
            for _ in range(height)
        ]

    def is_in_bounds(self, x: int, y: int) -> bool:
        # valid coordinates start at zero and stop before the width and height
        return 0 <= x < self.width and 0 <= y < self.height

    def get_cell(self, x: int, y: int) -> CellType:
        # check bounds before converting an x, y coordinate to row, column indexing
        if not self.is_in_bounds(x, y):
            raise IndexError("Grid coordinates are out of bounds.")

        return self.cells[y][x]
    
    def set_cell(self, x: int, y: int, cell_type: CellType) -> None:
        # update a cell within bounds; building rules such as overlap are handled separately
        if not self.is_in_bounds(x, y):
            raise IndexError("Grid coordinates are out of bounds.")

        self.cells[y][x] = cell_type
