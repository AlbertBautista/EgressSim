# defines an evacuation exit by its unique id and the grid cells it occupies
from dataclasses import dataclass
from .coordinate import Coordinate

@dataclass(frozen=True)
class Exit:

    id: str
    cells: tuple[Coordinate, ...]

    def __post_init__(self) -> None:
        # validate the id and cell list after creation; building checks placement and connectivity
        if not self.id.strip():
            raise ValueError("Exit id must not be empty.")

        if not self.cells:
            raise ValueError("Exit must contain at least one cell.")

        if len(set(self.cells)) != len(self.cells):
            raise ValueError("Exit cells must be unique.")
