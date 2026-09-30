# tests the final metrics generated from completed simulation runs
import pytest

from backend.simulation.agent import AgentSpec
from backend.simulation.building import Building
from backend.simulation.exit import Exit
from backend.simulation.scenario import Scenario
from backend.simulation.simulation import Simulation


# verifies a completed evacuation reports counts, exit usage, time, and run metadata
def test_result_reports_completed_evacuation():
    building = Building(7, 3)

    building.add_exit(
        Exit(
            id="exit_left",
            cells=((0, 1),),
        )
    )

    building.add_exit(
        Exit(
            id="exit_right",
            cells=((6, 1),),
        )
    )

    agents = (
        AgentSpec(
            id="agent_1",
            start_position=(1, 1),
            movement_speed=2.0,
            reaction_time=0.0,
            known_exit_ids=frozenset({"exit_left"}),
        ),
        AgentSpec(
            id="agent_2",
            start_position=(5, 1),
            movement_speed=2.0,
            reaction_time=0.0,
            known_exit_ids=frozenset({"exit_right"}),
        ),
    )

    scenario = Scenario(
        building=building,
        agent_specs=agents,
        hazard_cells=frozenset(),
        alarm_time=0.0,
    )

    simulation = Simulation(
        scenario=scenario,
        seed=42,
        timestep=0.5,
    )

    simulation.run(max_time=10.0)

    result = simulation.get_result()

    assert result.total_agents == 2
    assert result.evacuated_agents == 2
    assert result.remaining_agents == 0

    assert result.elapsed_time == pytest.approx(0.5)
    assert result.total_evacuation_time == pytest.approx(0.5)

    assert result.exit_usage == {
        "exit_left": 1,
        "exit_right": 1,
    }

    assert result.blocked_movement_attempts == 0
    assert result.termination_reason == "all_evacuated"
    assert result.seed == 42


# verifies an unfinished evacuation reports remaining agents and no total evacuation time
def test_result_reports_maximum_time_termination():
    building = Building(5, 3)

    building.add_exit(
        Exit(
            id="exit_1",
            cells=((4, 1),),
        )
    )

    agent = AgentSpec(
        id="agent_1",
        start_position=(1, 1),
        movement_speed=1.0,
        reaction_time=0.0,
        known_exit_ids=frozenset(),
    )

    scenario = Scenario(
        building=building,
        agent_specs=(agent,),
        hazard_cells=frozenset(),
        alarm_time=0.0,
    )

    simulation = Simulation(
        scenario=scenario,
        seed=42,
        timestep=0.5,
    )

    simulation.run(max_time=2.0)

    result = simulation.get_result()

    assert result.total_agents == 1
    assert result.evacuated_agents == 0
    assert result.remaining_agents == 1

    assert result.elapsed_time == pytest.approx(2.0)
    assert result.total_evacuation_time is None

    assert result.exit_usage == {
        "exit_1": 0,
    }

    assert result.termination_reason == "max_time"


# verifies final results cannot be requested before the simulation has terminated
def test_result_requires_terminated_simulation():
    building = Building(4, 3)

    agent = AgentSpec(
        id="agent_1",
        start_position=(1, 1),
        movement_speed=1.0,
        reaction_time=0.0,
        known_exit_ids=frozenset(),
    )

    scenario = Scenario(
        building=building,
        agent_specs=(agent,),
        hazard_cells=frozenset(),
    )

    simulation = Simulation(
        scenario=scenario,
        seed=42,
    )

    with pytest.raises(RuntimeError):
        simulation.get_result()


# verifies results are available when manual stepping completes an evacuation
def test_result_available_after_manual_step_completion():
    building = Building(4, 3)

    building.add_exit(
        Exit(
            id="exit_1",
            cells=((2, 1),),
        )
    )

    agent = AgentSpec(
        id="agent_1",
        start_position=(1, 1),
        movement_speed=2.0,
        reaction_time=0.0,
        known_exit_ids=frozenset({"exit_1"}),
    )

    scenario = Scenario(
        building=building,
        agent_specs=(agent,),
        hazard_cells=frozenset(),
        alarm_time=0.0,
    )

    simulation = Simulation(
        scenario=scenario,
        seed=42,
        timestep=0.5,
    )

    simulation.step()

    result = simulation.get_result()

    assert result.termination_reason == "all_evacuated"
    assert result.evacuated_agents == 1
    assert result.elapsed_time == pytest.approx(0.5)