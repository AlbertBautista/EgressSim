# tests simulation initialization, occupancy tracking, move application, and agent evacuation
import pytest

from backend.simulation.agent import AgentSpec
from backend.simulation.building import Building
from backend.simulation.scenario import Scenario
from backend.simulation.simulation import Simulation
from backend.simulation.exit import Exit
from backend.simulation.movement import MoveProposal


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


# verifies an accepted move updates both agent position and occupancy
def test_simulation_applies_accepted_move():
    simulation = Simulation(
        scenario=create_scenario(),
        seed=42,
    )

    resolution = simulation.move_agents(
        (
            MoveProposal(
                agent_id="agent_1",
                destination=(2, 1),
            ),
        )
    )

    agent = simulation.get_agent("agent_1")

    assert len(resolution.accepted) == 1
    assert resolution.blocked == ()

    assert agent.position == (2, 1)

    assert not simulation.is_occupied(1, 1)
    assert simulation.get_occupant_at(2, 1) is agent


# verifies moving into a cell occupied at the start of the phase is blocked
def test_simulation_does_not_apply_blocked_move():
    building = Building(4, 3)

    scenario = Scenario(
        building=building,
        agent_specs=(
            AgentSpec(
                id="agent_1",
                start_position=(1, 1),
                movement_speed=1.0,
                reaction_time=0.0,
                known_exit_ids=frozenset(),
            ),
            AgentSpec(
                id="agent_2",
                start_position=(2, 1),
                movement_speed=1.0,
                reaction_time=0.0,
                known_exit_ids=frozenset(),
            ),
        ),
        hazard_cells=frozenset(),
    )

    simulation = Simulation(
        scenario=scenario,
        seed=42,
    )

    proposal = MoveProposal(
        agent_id="agent_1",
        destination=(2, 1),
    )

    resolution = simulation.move_agents((proposal,))

    assert resolution.accepted == ()
    assert resolution.blocked == (proposal,)

    assert simulation.get_agent("agent_1").position == (1, 1)
    assert simulation.get_agent("agent_2").position == (2, 1)

    assert simulation.get_occupant_at(1, 1).id == "agent_1"
    assert simulation.get_occupant_at(2, 1).id == "agent_2"

    
# verifies entering an exit evacuates the agent and removes active occupancy
def test_simulation_evacuates_agent_entering_exit():
    building = Building(4, 3)

    building.add_exit(
        Exit(
            id="exit_1",
            cells=((2, 1),),
        )
    )

    scenario = Scenario(
        building=building,
        agent_specs=(
            AgentSpec(
                id="agent_1",
                start_position=(1, 1),
                movement_speed=1.0,
                reaction_time=0.0,
                known_exit_ids=frozenset({"exit_1"}),
            ),
        ),
        hazard_cells=frozenset(),
    )

    simulation = Simulation(
        scenario=scenario,
        seed=42,
    )

    simulation.move_agents(
        (
            MoveProposal(
                agent_id="agent_1",
                destination=(2, 1),
            ),
        )
    )

    agent = simulation.get_agent("agent_1")

    assert agent.position == (2, 1)
    assert agent.evacuated
    assert agent.evacuated_exit_id == "exit_1"

    assert not simulation.is_occupied(1, 1)
    assert not simulation.is_occupied(2, 1)


# verifies movement rejects destinations that are not adjacent or traversable
def test_simulation_rejects_invalid_move_proposals():
    building = Building(5, 4)

    building.place_wall(2, 1)

    scenario = Scenario(
        building=building,
        agent_specs=(
            AgentSpec(
                id="agent_1",
                start_position=(1, 1),
                movement_speed=1.0,
                reaction_time=0.0,
                known_exit_ids=frozenset(),
            ),
        ),
        hazard_cells=frozenset({(1, 2)}),
    )

    simulation = Simulation(
        scenario=scenario,
        seed=42,
    )

    with pytest.raises(ValueError):
        simulation.move_agents(
            (
                MoveProposal(
                    agent_id="agent_1",
                    destination=(3, 1),
                ),
            )
        )

    with pytest.raises(ValueError):
        simulation.move_agents(
            (
                MoveProposal(
                    agent_id="agent_1",
                    destination=(2, 1),
                ),
            )
        )

    with pytest.raises(ValueError):
        simulation.move_agents(
            (
                MoveProposal(
                    agent_id="agent_1",
                    destination=(1, 2),
                ),
            )
        )
