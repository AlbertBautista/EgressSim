import pytest

from backend.simulation.agent import AgentSpec
from backend.simulation.building import Building
from backend.simulation.exit import Exit
from backend.simulation.scenario import Scenario


def create_building():
    building = Building(6, 5)

    building.place_wall(2, 2)

    building.add_exit(
        Exit(
            id="exit_1",
            cells=((5, 2),)
        )
    )

    return building


def create_agent(
    agent_id="agent_1",
    start_position=(1, 1),
    known_exit_ids=frozenset({"exit_1"}),
):
    return AgentSpec(
        id=agent_id,
        start_position=start_position,
        movement_speed=1.0,
        reaction_time=2.0,
        known_exit_ids=known_exit_ids,
    )


def test_scenario_creation_and_traversability():
    # allow empty cells and exits, but block walls, hazards, and coordinates outside the building
    building = create_building()

    scenario = Scenario(
        building=building,
        agent_specs=(create_agent(),),
        hazard_cells=frozenset({(3, 2)}),
        alarm_time=5.0,
    )

    assert scenario.is_traversable(1, 1)
    assert not scenario.is_traversable(2, 2)
    assert not scenario.is_traversable(3, 2)
    assert scenario.is_traversable(5, 2)
    assert not scenario.is_traversable(6, 2)


def test_hazard_can_block_exit():
    # allow a hazard on an exit cell and make that cell unavailable for traversal
    building = create_building()

    scenario = Scenario(
        building=building,
        agent_specs=(create_agent(),),
        hazard_cells=frozenset({(5, 2)}),
        alarm_time=0.0,
    )

    assert not scenario.is_traversable(5, 2)


@pytest.mark.parametrize(
    "hazard_cell",
    [
        (2, 2),
        (6, 1),
    ],
)
def test_scenario_rejects_invalid_hazard_placement(hazard_cell):
    # reject hazards on walls or outside the building
    building = create_building()

    with pytest.raises(ValueError):
        Scenario(
            building=building,
            agent_specs=(create_agent(),),
            hazard_cells=frozenset({hazard_cell}),
        )


@pytest.mark.parametrize(
    "start_position",
    [
        (2, 2),
        (3, 2),
        (5, 2),
        (6, 1),
    ],
)
def test_scenario_rejects_invalid_agent_starting_positions(
    start_position,
):
    # reject agents starting on walls, hazards, exits, or outside the building
    building = create_building()

    with pytest.raises(ValueError):
        Scenario(
            building=building,
            agent_specs=(
                create_agent(
                    start_position=start_position
                ),
            ),
            hazard_cells=frozenset({(3, 2)}),
        )


def test_scenario_rejects_duplicate_agent_ids_and_positions():
    # reject shared agent ids even at different positions, and shared positions even with different ids
    building = create_building()

    with pytest.raises(ValueError):
        Scenario(
            building=building,
            agent_specs=(
                create_agent(
                    agent_id="agent_1",
                    start_position=(0, 0)
                ),
                create_agent(
                    agent_id="agent_1",
                    start_position=(1, 0)
                ),
            ),
            hazard_cells=frozenset(),
        )

    with pytest.raises(ValueError):
        Scenario(
            building=building,
            agent_specs=(
                create_agent(
                    agent_id="agent_1",
                    start_position=(0, 0)
                ),
                create_agent(
                    agent_id="agent_2",
                    start_position=(0, 0)
                ),
            ),
            hazard_cells=frozenset(),
        )


def test_scenario_rejects_unknown_exit_knowledge():
    # reject an agent whose known exit ids include an exit absent from the building
    building = create_building()

    with pytest.raises(ValueError):
        Scenario(
            building=building,
            agent_specs=(
                create_agent(
                    known_exit_ids=frozenset({"missing_exit"})
                ),
            ),
            hazard_cells=frozenset(),
        )


def test_scenario_rejects_negative_alarm_time():
    # reject an alarm scheduled before simulation time zero
    building = create_building()

    with pytest.raises(ValueError):
        Scenario(
            building=building,
            agent_specs=(create_agent(),),
            hazard_cells=frozenset(),
            alarm_time=-1.0,
        )