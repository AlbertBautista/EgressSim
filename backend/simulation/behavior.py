# defines the core timing rules that control when agents can evacuate and move
from .agent import AgentState
from .coordinate import Coordinate
from .pathfinding import find_path
from .scenario import Scenario


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