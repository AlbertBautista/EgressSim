# tests agent reaction and movement timing before simulation-step integration
import pytest

from backend.simulation.agent import AgentSpec, AgentState
from backend.simulation.behavior import (
    can_attempt_move,
    consume_movement_attempt,
    is_ready_to_evacuate,
    update_movement_progress,
)


def create_agent(
    movement_speed: float = 1.0,
    reaction_time: float = 0.0,
) -> AgentState:
    spec = AgentSpec(
        id="agent_1",
        start_position=(1, 1),
        movement_speed=movement_speed,
        reaction_time=reaction_time,
        known_exit_ids=frozenset(),
    )

    return AgentState.from_spec(spec)


# verifies an agent waits for both the alarm and its own reaction time
def test_agent_reaction_timing():
    agent = create_agent(reaction_time=2.0)

    assert not is_ready_to_evacuate(
        agent,
        current_time=5.0,
        alarm_activated_at=None,
    )

    assert not is_ready_to_evacuate(
        agent,
        current_time=6.5,
        alarm_activated_at=5.0,
    )

    assert is_ready_to_evacuate(
        agent,
        current_time=7.0,
        alarm_activated_at=5.0,
    )


# verifies movement speed builds progress and preserves unused fractional credit
def test_movement_progress():
    agent = create_agent(movement_speed=1.5)

    update_movement_progress(
        agent,
        timestep=0.5,
    )

    assert agent.movement_progress == pytest.approx(0.75)
    assert not can_attempt_move(agent)

    update_movement_progress(
        agent,
        timestep=0.5,
    )

    assert agent.movement_progress == pytest.approx(1.5)
    assert can_attempt_move(agent)

    consume_movement_attempt(agent)

    assert agent.movement_progress == pytest.approx(0.5)
    assert not can_attempt_move(agent)


# verifies movement credit cannot be consumed before an agent is eligible to move
def test_cannot_consume_insufficient_movement_progress():
    agent = create_agent(movement_speed=1.0)

    update_movement_progress(
        agent,
        timestep=0.5,
    )

    with pytest.raises(RuntimeError):
        consume_movement_attempt(agent)