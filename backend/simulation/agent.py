# defines the static configuration and changing runtime state of an agent
from dataclasses import dataclass

from .coordinate import Coordinate


@dataclass(frozen=True)
class AgentSpec:

    id: str
    start_position: Coordinate
    movement_speed: float
    reaction_time: float
    known_exit_ids: frozenset[str]
    communication_likelihood: float = 1.0

    def __post_init__(self) -> None:
        # validate attributes that describe how an agent begins a scenario
        if not self.id.strip():
            raise ValueError("Agent id must not be empty.")

        if self.movement_speed <= 0:
            raise ValueError("Agent movement speed must be positive.")

        if self.reaction_time < 0:
            raise ValueError("Agent reaction time cannot be negative.")

        if not 0.0 <= self.communication_likelihood <= 1.0:
            raise ValueError("Agent communication likelihood must be between 0 and 1.")


@dataclass
class AgentState:
    
    spec: AgentSpec
    position: Coordinate
    known_exit_ids: set[str]
    target_exit_id: str | None = None
    path: tuple[Coordinate, ...] = ()
    movement_progress: float = 0.0
    evacuated: bool = False
    evacuated_exit_id: str | None = None
    route_needs_update: bool = True

    @classmethod
    def from_spec(cls, spec: AgentSpec) -> "AgentState":
        # create fresh mutable state from the agent's scenario definition
        return cls(
            spec=spec,
            position=spec.start_position,
            known_exit_ids=set(spec.known_exit_ids),
        )

    @property
    def id(self) -> str:
        return self.spec.id