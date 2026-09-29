# defines the core timing rules that control when agents can evacuate and move
from .agent import AgentState


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