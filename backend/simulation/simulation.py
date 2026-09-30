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
from .behavior import (
    advance_route,
    consume_movement_attempt,
    create_move_proposal,
    is_ready_to_evacuate,
    plan_route,
    share_exit_knowledge,
    update_movement_progress,
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
        
        for spec in scenario.agent_specs:
            if spec.movement_speed * timestep > 1.0:
                raise ValueError(
                    "Agent movement speed is too high for the simulation timestep."
                )

        self.scenario = scenario
        self.seed = seed
        self.timestep = timestep
        self.time = 0.0

        self._alarm_activated_at: float | None = None
        self._termination_reason: str | None = None

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


    @property
    def alarm_activated_at(self) -> float | None:
        return self._alarm_activated_at

    @property
    def alarm_active(self) -> bool:
        return self._alarm_activated_at is not None

    def trigger_alarm(self) -> None:
        # allow a manual alarm to activate at the current simulation time
        if self._alarm_activated_at is None:
            self._alarm_activated_at = self.time

    def _activate_scheduled_alarm_if_due(self) -> None:
        if self._alarm_activated_at is not None:
            return

        alarm_time = self.scenario.alarm_time

        if alarm_time is None:
            return

        if self.time >= alarm_time:
            self._alarm_activated_at = alarm_time
    

    def step(self) -> MovementResolution:
        # process one fixed interval of simulation behavior and movement
        self._activate_scheduled_alarm_if_due()

        if not self.alarm_active:
            self.time += self.timestep

            return MovementResolution(
                accepted=(),
                blocked=(),
            )

        active_agents = tuple(
            agent
            for agent in self._agents.values()
            if not agent.evacuated
        )

        ready_agents = tuple(
            agent
            for agent in active_agents
            if is_ready_to_evacuate(
                agent=agent,
                current_time=self.time,
                alarm_activated_at=self._alarm_activated_at,
            )
        )

        changed_agent_ids = share_exit_knowledge(
            agents=ready_agents,
            rng=self._rng,
        )

        # newly learned exit information may change an agent's best route
        for agent_id in changed_agent_ids:
            self._agents[agent_id].route_needs_update = True

        proposals: list[MoveProposal] = []

        for agent in ready_agents:
            if agent.route_needs_update:
                plan_route(
                    agent=agent,
                    scenario=self.scenario,
                )

                agent.route_needs_update = False

            # no usable route means no movement progress should accumulate
            if len(agent.path) < 2:
                continue

            update_movement_progress(
                agent=agent,
                timestep=self.timestep,
            )

            proposal = create_move_proposal(agent)

            if proposal is None:
                continue

            # attempting movement consumes credit whether the move succeeds or fails
            consume_movement_attempt(agent)

            proposals.append(proposal)

        resolution = self.move_agents(tuple(proposals))

        for proposal in resolution.accepted:
            agent = self._agents[proposal.agent_id]

            if not agent.evacuated:
                advance_route(
                    agent=agent,
                    destination=proposal.destination,
                )

        self.time += self.timestep

        return resolution


    @property
    def all_agents_evacuated(self) -> bool:
        return all(
            agent.evacuated
            for agent in self._agents.values()
        )


    @property
    def termination_reason(self) -> str | None:
        return self._termination_reason
    

    def run(self, max_time: float = 300.0) -> None:
        # run complete simulation steps until everyone evacuates or the time limit is reached
        if max_time <= 0:
            raise ValueError("Maximum simulation time must be positive.")

        self._termination_reason = None

        if self.all_agents_evacuated:
            self._termination_reason = "all_evacuated"
            return

        while self.time + self.timestep <= max_time:
            self.step()

            if self.all_agents_evacuated:
                self._termination_reason = "all_evacuated"
                return

        self._termination_reason = "max_time"