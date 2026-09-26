# manages the mutable state for one simulation run
import random

from .agent import AgentState
from .coordinate import Coordinate
from .scenario import Scenario


class Simulation:

    def __init__(
        self,
        scenario: Scenario,
        seed: int,
        timestep: float = 0.5,
    ):
        if timestep <= 0:
            raise ValueError("Simulation timestep must be positive.")

        self.scenario = scenario
        self.seed = seed
        self.timestep = timestep
        self.time = 0.0

        # one seeded random source keeps runtime randomness reproducible
        self._rng = random.Random(seed)

        # every run receives fresh state instead of modifying the scenario
        self._agents = {
            spec.id: AgentState.from_spec(spec)
            for spec in scenario.agent_specs
        }

        # occupancy tracks which active agent currently owns each grid cell
        self._occupancy: dict[Coordinate, str] = {
            state.position: state.id
            for state in self._agents.values()
        }

    @property
    def agents(self) -> tuple[AgentState, ...]:
        return tuple(self._agents.values())

    def get_agent(self, agent_id: str) -> AgentState:
        return self._agents[agent_id]

    def is_occupied(self, x: int, y: int) -> bool:
        return (x, y) in self._occupancy

    def get_occupant_at(
        self,
        x: int,
        y: int,
    ) -> AgentState | None:
        agent_id = self._occupancy.get((x, y))

        if agent_id is None:
            return None

        return self._agents[agent_id]