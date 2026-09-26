import pytest

from backend.simulation.agent import AgentSpec
from backend.simulation.agent import AgentSpec, AgentState


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


def test_agent_state_is_created_from_spec():
    # check that runtime state uses the spec's position and exits, with no target, path, or movement progress
    spec = AgentSpec(
        id="agent_1",
        start_position=(2, 3),
        movement_speed=1.2,
        reaction_time=3.0,
        known_exit_ids=frozenset({"exit_1"}),
    )

    state = AgentState.from_spec(spec)

    assert state.id == "agent_1"
    assert state.spec == spec
    assert state.position == (2, 3)
    assert state.known_exit_ids == {"exit_1"}
    assert state.target_exit_id is None
    assert state.path == ()
    assert state.movement_progress == 0.0


def test_agent_state_knowledge_does_not_modify_spec():
    # check that learning an exit during a run leaves the spec's initial exit knowledge unchanged
    spec = AgentSpec(
        id="agent_1",
        start_position=(2, 3),
        movement_speed=1.0,
        reaction_time=0.0,
        known_exit_ids=frozenset({"exit_1"}),
    )

    state = AgentState.from_spec(spec)

    state.known_exit_ids.add("exit_2")

    assert state.known_exit_ids == {"exit_1", "exit_2"}
    assert spec.known_exit_ids == frozenset({"exit_1"})
