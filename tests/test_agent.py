import pytest

from backend.simulation.agent import AgentSpec


def test_agent_spec_creation():
    # check that an agent stores its starting configuration, including no known exits
    agent = AgentSpec(
        id="agent_1",
        start_position=(2, 3),
        movement_speed=1.2,
        reaction_time=3.0,
        known_exit_ids=frozenset()
    )

    assert agent.id == "agent_1"
    assert agent.start_position == (2, 3)
    assert agent.movement_speed == 1.2
    assert agent.reaction_time == 3.0
    assert agent.known_exit_ids == frozenset()


@pytest.mark.parametrize(
    ("agent_id", "movement_speed", "reaction_time"),
    [
        ("", 1.0, 0.0),
        ("agent_1", 0.0, 0.0),
        ("agent_1", -1.0, 0.0),
        ("agent_1", 1.0, -1.0),
    ],
)
def test_agent_spec_rejects_invalid_attributes(
    agent_id,
    movement_speed,
    reaction_time,
):
    # reject an empty id, zero or negative movement speed, and negative reaction time
    with pytest.raises(ValueError):
        AgentSpec(
            id=agent_id,
            start_position=(1, 1),
            movement_speed=movement_speed,
            reaction_time=reaction_time,
            known_exit_ids=frozenset()
        )