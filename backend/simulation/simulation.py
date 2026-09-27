# manages the mutable state for one simulation run
import random

from .agent import AgentState
from .coordinate import Coordinate
from .scenario import Scenario
from .movement import (
    MoveProposal,
    MovementResolution,
    resolve_move_conflicts,
)


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

    def move_agents(
        self,
        proposals: tuple[MoveProposal, ...],
    ) -> MovementResolution:
        # validate proposed moves before conflict resolution changes runtime state
        self._validate_move_proposals(proposals)

        resolution = resolve_move_conflicts(
            proposals=proposals,
            occupied_cells=frozenset(self._occupancy),
            rng=self._rng,
        )

        # remove accepted agents from their old cells before applying new positions
        for proposal in resolution.accepted:
            agent = self._agents[proposal.agent_id]
            del self._occupancy[agent.position]

        for proposal in resolution.accepted:
            agent = self._agents[proposal.agent_id]

            agent.position = proposal.destination

            building_exit = self.scenario.building.get_exit_at(
                *proposal.destination
            )

            if building_exit is not None:
                agent.evacuated = True
                agent.evacuated_exit_id = building_exit.id
                agent.target_exit_id = None
                agent.path = ()
                continue

            self._occupancy[proposal.destination] = agent.id

        return resolution

    def _validate_move_proposals(
        self,
        proposals: tuple[MoveProposal, ...],
    ) -> None:
        seen_agent_ids: set[str] = set()

        for proposal in proposals:
            if proposal.agent_id in seen_agent_ids:
                raise ValueError(
                    "Each agent may submit only one move proposal."
                )

            seen_agent_ids.add(proposal.agent_id)

            if proposal.agent_id not in self._agents:
                raise ValueError("Move proposal references an unknown agent.")

            agent = self._agents[proposal.agent_id]

            if agent.evacuated:
                raise ValueError("Evacuated agents cannot move.")

            # position and occupancy must remain synchronized throughout a run
            if self._occupancy.get(agent.position) != agent.id:
                raise RuntimeError(
                    "Agent position and occupancy are inconsistent."
                )

            x1, y1 = agent.position
            x2, y2 = proposal.destination

            if abs(x1 - x2) + abs(y1 - y2) != 1:
                raise ValueError(
                    "Agents may only move one cell at a time."
                )

            if not self.scenario.is_traversable(x2, y2):
                raise ValueError(
                    "Agents cannot move into a non-traversable cell."
                )