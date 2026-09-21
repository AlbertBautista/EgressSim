from enum import Enum

# defines the permanent types of cells that can exist in a building grid
class CellType(Enum):
    EMPTY = "empty"
    WALL = "wall"
    EXIT = "exit"