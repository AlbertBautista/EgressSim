from enum import Enum

class CellType(Enum):
    EMPTY = "empty"
    WALL = "wall"
    EXIT = "exit"
    HAZARD = "hazard"