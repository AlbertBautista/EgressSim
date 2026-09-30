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


# verifies one simulation step coordinates routing, movement, and time progression
def test_step_moves_ready_agent():
    building = Building(5, 3)

    building.add_exit(
        Exit(
            id="exit_1",
            cells=((4, 1),),
        )
    )

    spec = AgentSpec(
        id="agent_1",
        start_position=(1, 1),
        movement_speed=2.0,
        reaction_time=0.0,
        known_exit_ids=frozenset({"exit_1"}),
    )

    scenario = Scenario(
        building=building,
        agent_specs=(spec,),
        hazard_cells=frozenset(),
        alarm_time=0.0,
    )

    simulation = Simulation(
        scenario=scenario,
        seed=42,
        timestep=0.5,
    )

    simulation.step()

    agent = simulation.get_agent("agent_1")

    assert agent.position == (2, 1)
    assert agent.path[0] == (2, 1)
    assert simulation.time == pytest.approx(0.5)


# verifies an agent remains stationary until its reaction delay has passed
def test_step_respects_agent_reaction_time():
    building = Building(5, 3)

    building.add_exit(
        Exit(
            id="exit_1",
            cells=((4, 1),),
        )
    )

    spec = AgentSpec(
        id="agent_1",
        start_position=(1, 1),
        movement_speed=2.0,
        reaction_time=1.0,
        known_exit_ids=frozenset({"exit_1"}),
    )

    scenario = Scenario(
        building=building,
        agent_specs=(spec,),
        hazard_cells=frozenset(),
        alarm_time=0.0,
    )

    simulation = Simulation(
        scenario=scenario,
        seed=42,
        timestep=0.5,
    )

    simulation.step()
    simulation.step()

    assert simulation.get_agent("agent_1").position == (1, 1)

    simulation.step()

    assert simulation.get_agent("agent_1").position == (2, 1)


# verifies an agent cannot accumulate movement credit while waiting without a route
def test_step_does_not_accumulate_progress_without_route():
    building = Building(5, 3)

    spec = AgentSpec(
        id="agent_1",
        start_position=(1, 1),
        movement_speed=2.0,
        reaction_time=0.0,
        known_exit_ids=frozenset(),
    )

    scenario = Scenario(
        building=building,
        agent_specs=(spec,),
        hazard_cells=frozenset(),
        alarm_time=0.0,
    )

    simulation = Simulation(
        scenario=scenario,
        seed=42,
        timestep=0.5,
    )

    simulation.step()
    simulation.step()
    simulation.step()

    agent = simulation.get_agent("agent_1")

    assert agent.position == (1, 1)
    assert agent.movement_progress == pytest.approx(0.0)


# verifies a scenario without a scheduled alarm can be started manually
def test_manual_alarm_starts_evacuation():
    building = Building(5, 3)

    building.add_exit(
        Exit(
            id="exit_1",
            cells=((4, 1),),
        )
    )

    spec = AgentSpec(
        id="agent_1",
        start_position=(1, 1),
        movement_speed=2.0,
        reaction_time=0.0,
        known_exit_ids=frozenset({"exit_1"}),
    )

    scenario = Scenario(
        building=building,
        agent_specs=(spec,),
        hazard_cells=frozenset(),
        alarm_time=None,
    )

    simulation = Simulation(
        scenario=scenario,
        seed=42,
        timestep=0.5,
    )

    simulation.step()

    assert simulation.get_agent("agent_1").position == (1, 1)

    simulation.trigger_alarm()
    simulation.step()

    assert simulation.get_agent("agent_1").position == (2, 1)


# verifies the timestep rejects speeds that cannot be represented by one move per tick
def test_simulation_rejects_speed_too_high_for_timestep():
    building = Building(4, 3)

    spec = AgentSpec(
        id="agent_1",
        start_position=(1, 1),
        movement_speed=2.1,
        reaction_time=0.0,
        known_exit_ids=frozenset(),
    )

    scenario = Scenario(
        building=building,
        agent_specs=(spec,),
        hazard_cells=frozenset(),
    )

    with pytest.raises(ValueError):
        Simulation(
            scenario=scenario,
            seed=42,
            timestep=0.5,
        )


# verifies run continues stepping until every agent has evacuated
def test_run_stops_when_all_agents_evacuate():
    building = Building(5, 3)

    building.add_exit(
        Exit(
            id="exit_1",
            cells=((3, 1),),
        )
    )

    spec = AgentSpec(
        id="agent_1",
        start_position=(1, 1),
        movement_speed=2.0,
        reaction_time=0.0,
        known_exit_ids=frozenset({"exit_1"}),
    )

    scenario = Scenario(
        building=building,
        agent_specs=(spec,),
        hazard_cells=frozenset(),
        alarm_time=0.0,
    )

    simulation = Simulation(
        scenario=scenario,
        seed=42,
        timestep=0.5,
    )

    simulation.run(max_time=10.0)

    agent = simulation.get_agent("agent_1")

    assert agent.evacuated
    assert agent.evacuated_exit_id == "exit_1"
    assert simulation.all_agents_evacuated
    assert simulation.termination_reason == "all_evacuated"
    assert simulation.time == pytest.approx(1.0)


# verifies run stops at the time limit when evacuation cannot finish
def test_run_stops_at_maximum_time():
    building = Building(5, 3)

    spec = AgentSpec(
        id="agent_1",
        start_position=(1, 1),
        movement_speed=1.0,
        reaction_time=0.0,
        known_exit_ids=frozenset(),
    )

    scenario = Scenario(
        building=building,
        agent_specs=(spec,),
        hazard_cells=frozenset(),
        alarm_time=0.0,
    )

    simulation = Simulation(
        scenario=scenario,
        seed=42,
        timestep=0.5,
    )

    simulation.run(max_time=2.0)

    assert not simulation.all_agents_evacuated
    assert simulation.termination_reason == "max_time"
    assert simulation.time == pytest.approx(2.0)


# verifies run rejects a non-positive simulation time limit
def test_run_rejects_invalid_maximum_time():
    simulation = Simulation(
        scenario=create_scenario(),
        seed=42,
    )

    with pytest.raises(ValueError):
        simulation.run(max_time=0.0)

    with pytest.raises(ValueError):
        simulation.run(max_time=-1.0)


# verifies agents do not communicate until their reaction delays have passed
def test_step_respects_reaction_time_for_communication():
    building = Building(5, 3)

    building.add_exit(
        Exit(
            id="exit_1",
            cells=((4, 1),),
        )
    )

    informed_agent = AgentSpec(
        id="agent_1",
        start_position=(1, 1),
        movement_speed=0.1,
        reaction_time=0.0,
        known_exit_ids=frozenset({"exit_1"}),
        communication_likelihood=1.0,
    )

    waiting_agent = AgentSpec(
        id="agent_2",
        start_position=(2, 1),
        movement_speed=0.1,
        reaction_time=2.0,
        known_exit_ids=frozenset(),
        communication_likelihood=1.0,
    )

    scenario = Scenario(
        building=building,
        agent_specs=(informed_agent, waiting_agent),
        hazard_cells=frozenset(),
        alarm_time=0.0,
    )

    simulation = Simulation(
        scenario=scenario,
        seed=42,
        timestep=0.5,
    )

    simulation.step()

    assert simulation.get_agent("agent_2").known_exit_ids == set()


# verifies blocked movement conflicts are counted as congestion
def test_step_tracks_blocked_movement_attempts():
    building = Building(5, 4)

    building.place_wall(0, 0)
    building.place_wall(2, 0)

    building.place_wall(0, 2)
    building.place_wall(2, 2)
    building.place_wall(1, 3)

    building.add_exit(
        Exit(
            id="exit_1",
            cells=((4, 1),),
        )
    )

    agents = (
        AgentSpec(
            id="agent_1",
            start_position=(1, 0),
            movement_speed=2.0,
            reaction_time=0.0,
            known_exit_ids=frozenset({"exit_1"}),
        ),
        AgentSpec(
            id="agent_2",
            start_position=(1, 2),
            movement_speed=2.0,
            reaction_time=0.0,
            known_exit_ids=frozenset({"exit_1"}),
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

    simulation.step()

    assert simulation.blocked_movement_attempts == 1