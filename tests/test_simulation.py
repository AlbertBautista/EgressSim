import pytest

from backend.simulation.agent import AgentSpec
from backend.simulation.building import Building
from backend.simulation.scenario import Scenario
from backend.simulation.simulation import Simulation


def create_scenario():
    building = Building(5, 4)

    agents = (
        AgentSpec(
            id="agent_1",
            start_position=(1, 1),
            movement_speed=1.0,
            reaction_time=0.0,
            known_exit_ids=frozenset(),
        ),
        AgentSpec(
            id="agent_2",
            start_position=(3, 2),
            movement_speed=1.2,
            reaction_time=2.0,
            known_exit_ids=frozenset(),
        ),
    )

    return Scenario(
        building=building,
        agent_specs=agents,
        hazard_cells=frozenset(),
    )


def test_simulation_creation():
    # check that a run stores its scenario and seed, starts at time zero, and creates both agents
    # the default timestep is 0.5 simulation seconds
    scenario = create_scenario()

    simulation = Simulation(
        scenario=scenario,
        seed=42,
    )

    assert simulation.scenario is scenario
    assert simulation.seed == 42
    assert simulation.timestep == 0.5
    assert simulation.time == 0.0
    assert len(simulation.agents) == 2


def test_simulation_creates_fresh_agent_states():
    # check that changing one run's agent position and knowledge leaves another run and the specs unchanged
    scenario = create_scenario()

    first_run = Simulation(
        scenario=scenario,
        seed=42,
    )

    second_run = Simulation(
        scenario=scenario,
        seed=42,
    )

    first_agent = first_run.get_agent("agent_1")
    second_agent = second_run.get_agent("agent_1")

    first_agent.position = (2, 1)
    first_agent.known_exit_ids.add("new_exit")

    assert second_agent.position == (1, 1)
    assert "new_exit" not in second_agent.known_exit_ids

    assert scenario.agent_specs[0].start_position == (1, 1)
    assert scenario.agent_specs[0].known_exit_ids == frozenset()


def test_simulation_tracks_starting_occupancy():
    # check that starting cells map to the correct agents and an unoccupied cell has no occupant
    simulation = Simulation(
        scenario=create_scenario(),
        seed=42,
    )

    assert simulation.is_occupied(1, 1)
    assert simulation.is_occupied(3, 2)
    assert not simulation.is_occupied(0, 0)

    assert simulation.get_occupant_at(1, 1).id == "agent_1"
    assert simulation.get_occupant_at(3, 2).id == "agent_2"
    assert simulation.get_occupant_at(0, 0) is None


def test_simulation_rejects_invalid_timestep():
    # reject zero and negative timesteps so simulation time can advance
    with pytest.raises(ValueError):
        Simulation(
            scenario=create_scenario(),
            seed=42,
            timestep=0.0,
        )

    with pytest.raises(ValueError):
        Simulation(
            scenario=create_scenario(),
            seed=42,
            timestep=-0.5,
        )
