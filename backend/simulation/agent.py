# defines the starting configuration of an agent
from dataclasses import dataclass
from .coordinate import Coordinate


@dataclass(frozen=True)
class AgentSpec:
    id: str
    start_position: Coordinate
    movement_speed: float
    reaction_time: float
    known_exit_ids: frozenset[str]

    def __post_init__(self) -> None:
        # validate the id, speed, and reaction time after creation; scenario checks placement and exits
        if not self.id.strip():
            raise ValueError("Agent id must not be empty.")

        if self.movement_speed <= 0:
            raise ValueError("Agent movement speed must be positive.")

        if self.reaction_time < 0:
            raise ValueError("Agent reaction time cannot be negative.")
