# defines the core timing rules that control when agents can evacuate and move
from .agent import AgentState
from .coordinate import Coordinate
from .pathfinding import find_path
from .scenario import Scenario
from random import Random
from .movement import MoveProposal


def is_ready_to_evacuate(
    agent: AgentState,
    current_time: float,
    alarm_activated_at: float | None,
) -> bool:
    # an agent can react only after an alarm has activated and its reaction delay has passed
    if alarm_activated_at is None:
        return False

    if agent.evacuated:
        return False

    return current_time >= alarm_activated_at + agent.spec.reaction_time


def update_movement_progress(
    agent: AgentState,
    timestep: float,
) -> None:
    agent.movement_progress += (
        agent.spec.movement_speed * timestep
    )


def can_attempt_move(agent: AgentState) -> bool:
    return agent.movement_progress >= 1.0


def consume_movement_attempt(agent: AgentState) -> None:
    # consume one cell of movement credit whenever an agent attempts a move
    if not can_attempt_move(agent):
        raise RuntimeError(
            "Agent does not have enough movement progress."
        )

    agent.movement_progress -= 1.0


# finds a reachable known exit and stores the route the agent should follow
def plan_route(
    agent: AgentState,
    scenario: Scenario,
) -> bool:
    if agent.evacuated:
        return False

    goal_cells: set[Coordinate] = set()

    for exit_id in agent.known_exit_ids:
        building_exit = scenario.building.get_exit(exit_id)
        goal_cells.update(building_exit.cells)

    if not goal_cells:
        agent.target_exit_id = None
        agent.path = ()
        return False

    path = find_path(
        start=agent.position,
        goals=frozenset(goal_cells),
        is_traversable=scenario.is_traversable,
    )

    if path is None:
        agent.target_exit_id = None
        agent.path = ()
        return False

    reached_exit = scenario.building.get_exit_at(*path[-1])

    if reached_exit is None:
        raise RuntimeError(
            "Path ended at a goal that does not belong to an exit."
        )

    agent.target_exit_id = reached_exit.id
    agent.path = path

    return True


def share_exit_knowledge(
    agents: tuple[AgentState, ...],
    rng: Random,
) -> frozenset[str]:
    # snapshot knowledge so information cannot cascade through several agents in one phase
    ordered_agents = sorted(
        (agent for agent in agents if not agent.evacuated),
        key=lambda agent: agent.id,
    )

    knowledge_snapshot = {
        agent.id: frozenset(agent.known_exit_ids)
        for agent in ordered_agents
    }

    learned_exits: dict[str, set[str]] = {}

    for sender in ordered_agents:
        for recipient in ordered_agents:
            if sender.id == recipient.id:
                continue

            if not _are_adjacent(sender.position, recipient.position):
                continue

            new_exit_ids = (
                knowledge_snapshot[sender.id]
                - knowledge_snapshot[recipient.id]
            )

            if not new_exit_ids:
                continue

            if rng.random() >= sender.spec.communication_likelihood:
                continue

            learned_exits.setdefault(
                recipient.id,
                set(),
            ).update(new_exit_ids)

    for agent in ordered_agents:
        new_exit_ids = learned_exits.get(agent.id)

        if new_exit_ids:
            agent.known_exit_ids.update(new_exit_ids)

    return frozenset(learned_exits)


def _are_adjacent(
    first: Coordinate,
    second: Coordinate,
) -> bool:
    x1, y1 = first
    x2, y2 = second

    return abs(x1 - x2) + abs(y1 - y2) == 1


def create_move_proposal(
    agent: AgentState,
) -> MoveProposal | None:
    # create a proposal for the next route cell when the agent is ready to move
    if agent.evacuated:
        return None

    if not can_attempt_move(agent):
        return None

    if len(agent.path) < 2:
        return None

    if agent.path[0] != agent.position:
        raise RuntimeError(
            "Agent position and path are inconsistent."
        )

    return MoveProposal(
        agent_id=agent.id,
        destination=agent.path[1],
    )


def advance_route(
    agent: AgentState,
    destination: Coordinate,
) -> None:
    # remove the completed path step after an accepted move
    if agent.evacuated:
        return

    if len(agent.path) < 2:
        raise RuntimeError(
            "Agent does not have a route step to complete."
        )

    if agent.path[1] != destination:
        raise RuntimeError(
            "Accepted destination does not match the agent route."
        )

    if agent.position != destination:
        raise RuntimeError(
            "Agent position was not updated to the accepted destination."
        )

    agent.path = agent.path[1:]