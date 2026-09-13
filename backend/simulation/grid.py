from .cell import CellType

class Grid:
    def __init__(self, width: int, height: int):
        if width <= 0 or height <= 0:
            raise ValueError("Grid dimensions must be positive.")

        self.width = width
        self.height = height

        self.cells = [
            [CellType.EMPTY for _ in range(width)]
            for _ in range(height)
        ]

    def is_in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    def get_cell(self, x: int, y: int) -> CellType:
        if not self.is_in_bounds(x, y):
            raise IndexError("Grid coordinates are out of bounds.")

        return self.cells[y][x]
    
    def set_cell(self, x: int, y: int, cell_type: CellType) -> None:
        if not self.is_in_bounds(x, y):
            raise IndexError("Grid coordinates are out of bounds.")

        self.cells[y][x] = cell_type