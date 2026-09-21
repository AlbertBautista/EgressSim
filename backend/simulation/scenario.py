# combines a building with agents, hazards, and alarm timing to define an evacuation scenario
from dataclasses import dataclass

from .agent import AgentSpec
from .building import Building
from .cell import CellType
from .coordinate import Coordinate


@dataclass(frozen=True)
class Scenario:
    building: Building
    agent_specs: tuple[AgentSpec, ...]
    hazard_cells: frozenset[Coordinate]
    alarm_time: float | None = None

    def __post_init__(self) -> None:
        # validate alarm timing, hazard placement, and agent configuration after creation
        if self.alarm_time is not None and self.alarm_time < 0:
            raise ValueError("Alarm time cannot be negative.")

        self._validate_hazards()
        self._validate_agents()

    def is_traversable(self, x: int, y: int) -> bool:
        # combine building bounds, walls, and scenario hazards to determine whether a cell is usable

        if not self.building.is_in_bounds(x, y):
            return False

        if (x, y) in self.hazard_cells:
            return False

        return self.building.get_cell(x, y) != CellType.WALL

    def _validate_hazards(self) -> None:
        # require hazards to be inside the building and off walls; covering exit cells is allowed
        for x, y in self.hazard_cells:
            if not self.building.is_in_bounds(x, y):
                raise ValueError("Hazard cells must be inside the building.")

            if self.building.get_cell(x, y) == CellType.WALL:
                raise ValueError("Hazards cannot be placed on walls.")

    def _validate_agents(self) -> None:
        # require unique ids, distinct safe starting cells, and knowledge of existing exits only
        agent_ids: set[str] = set()
        starting_positions: set[Coordinate] = set()

        exit_ids = {
            building_exit.id
            for building_exit in self.building.exits
        }

        for agent in self.agent_specs:
            if agent.id in agent_ids:
                raise ValueError("Agent ids must be unique.")

            agent_ids.add(agent.id)

            x, y = agent.start_position

            if not self.building.is_in_bounds(x, y):
                raise ValueError(
                    "Agent starting positions must be inside the building."
                )

            if self.building.get_cell(x, y) != CellType.EMPTY:
                raise ValueError(
                    "Agents must start on empty building cells."
                )

            if agent.start_position in self.hazard_cells:
                raise ValueError(
                    "Agents cannot start on hazard cells."
                )

            if agent.start_position in starting_positions:
                raise ValueError(
                    "Agents cannot share a starting position."
                )

            starting_positions.add(agent.start_position)

            # set subtraction finds known exit ids that are missing from this building
            unknown_exit_ids = agent.known_exit_ids - exit_ids

            if unknown_exit_ids:
                raise ValueError(
                    "Agent known exits must exist in the building."
                )
