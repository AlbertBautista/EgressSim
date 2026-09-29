# tests agent reaction and movement timing before simulation-step integration
import pytest
from backend.simulation.building import Building
from backend.simulation.exit import Exit
from backend.simulation.scenario import Scenario
from backend.simulation.behavior import plan_route

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


# verifies an agent routes only toward exits it currently knows
def test_agent_plans_route_to_known_exit():
    building = Building(7, 3)

    building.add_exit(
        Exit(
            id="unknown_exit",
            cells=((2, 1),),
        )
    )

    building.add_exit(
        Exit(
            id="known_exit",
            cells=((6, 1),),
        )
    )

    spec = AgentSpec(
        id="agent_1",
        start_position=(0, 1),
        movement_speed=1.0,
        reaction_time=0.0,
        known_exit_ids=frozenset({"known_exit"}),
    )

    scenario = Scenario(
        building=building,
        agent_specs=(spec,),
        hazard_cells=frozenset(),
    )

    agent = AgentState.from_spec(spec)

    assert plan_route(agent, scenario)

    assert agent.target_exit_id == "known_exit"
    assert agent.path[0] == (0, 1)
    assert agent.path[-1] == (6, 1)


# verifies an agent chooses a reachable known exit when another known exit is blocked
def test_agent_uses_reachable_known_exit():
    building = Building(7, 5)

    building.add_exit(
        Exit(
            id="blocked_exit",
            cells=((3, 0),),
        )
    )

    building.add_exit(
        Exit(
            id="available_exit",
            cells=((6, 2),),
        )
    )

    spec = AgentSpec(
        id="agent_1",
        start_position=(0, 2),
        movement_speed=1.0,
        reaction_time=0.0,
        known_exit_ids=frozenset({
            "blocked_exit",
            "available_exit",
        }),
    )

    scenario = Scenario(
        building=building,
        agent_specs=(spec,),
        hazard_cells=frozenset({(3, 0)}),
    )

    agent = AgentState.from_spec(spec)

    assert plan_route(agent, scenario)

    assert agent.target_exit_id == "available_exit"
    assert agent.path[-1] == (6, 2)


# verifies an agent without a reachable known exit has no target or path
def test_agent_has_no_route_without_reachable_known_exit():
    building = Building(5, 3)

    building.add_exit(
        Exit(
            id="exit_1",
            cells=((4, 1),),
        )
    )

    spec = AgentSpec(
        id="agent_1",
        start_position=(0, 1),
        movement_speed=1.0,
        reaction_time=0.0,
        known_exit_ids=frozenset(),
    )

    scenario = Scenario(
        building=building,
        agent_specs=(spec,),
        hazard_cells=frozenset(),
    )

    agent = AgentState.from_spec(spec)

    assert not plan_route(agent, scenario)

    assert agent.target_exit_id is None
    assert agent.path == ()