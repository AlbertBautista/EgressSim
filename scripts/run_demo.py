# runs predefined egresssim scenarios as a visual terminal prototype
import os
import time

from backend.simulation.agent import AgentSpec
from backend.simulation.building import Building
from backend.simulation.cell import CellType
from backend.simulation.exit import Exit
from backend.simulation.scenario import Scenario
from backend.simulation.simulation import Simulation


DISPLAY_DELAY = 0.75
MAX_SIMULATION_TIME = 60.0

# stores the shared prototype population as id, position, speed, and reaction time
DEMO_AGENT_DATA = (
    ("agent_1", (2, 2), 1.0, 0.0),
    ("agent_2", (2, 3), 1.5, 0.5),
    ("agent_3", (5, 2), 2.0, 1.0),
    ("agent_4", (7, 2), 1.0, 1.5),
    ("agent_5", (8, 5), 1.5, 0.0),
    ("agent_6", (9, 5), 1.0, 0.5),
)


def create_demo_building() -> Building:
    # create the building layout shared by the prototype scenarios
    building = Building(12, 8)

    right_exit_cells = (
        (11, 2),
        (11, 3),
    )

    bottom_exit_cells = (
        (5, 7),
        (6, 7),
    )

    # place exterior walls while leaving space for the exits
    for x in range(building.width):
        building.place_wall(x, 0)

        if (x, 7) not in bottom_exit_cells:
            building.place_wall(x, 7)

    for y in range(1, building.height - 1):
        building.place_wall(0, y)

        if (11, y) not in right_exit_cells:
            building.place_wall(11, y)

    # create an interior barrier with two possible openings
    for x in range(3, 9):
        if x not in (4, 7):
            building.place_wall(x, 4)

    building.add_exit(
        Exit(
            id="exit_right",
            cells=right_exit_cells,
        )
    )

    building.add_exit(
        Exit(
            id="exit_bottom",
            cells=bottom_exit_cells,
        )
    )

    return building


def create_demo_agents(
    full_exit_knowledge: bool,
    communication_likelihood: float = 1.0,
) -> tuple[AgentSpec, ...]:
    # create the shared prototype population with the requested exit knowledge
    agents: list[AgentSpec] = []

    for agent_id, position, speed, reaction_time in DEMO_AGENT_DATA:
        if full_exit_knowledge:
            known_exit_ids = frozenset({
                "exit_right",
                "exit_bottom",
            })

        elif agent_id == "agent_5":
            known_exit_ids = frozenset({
                "exit_right",
                "exit_bottom",
            })

        else:
            known_exit_ids = frozenset({
                "exit_right",
            })

        agents.append(
            AgentSpec(
                id=agent_id,
                start_position=position,
                movement_speed=speed,
                reaction_time=reaction_time,
                known_exit_ids=known_exit_ids,
                communication_likelihood=communication_likelihood,
            )
        )

    return tuple(agents)


def create_demo_scenario() -> Scenario:
    building = create_demo_building()

    return Scenario(
        building=building,
        agent_specs=create_demo_agents(
            full_exit_knowledge=True,
        ),
        hazard_cells=frozenset({
            (4, 4),
        }),
        alarm_time=0.0,
    )


def create_comparison_scenario(
    full_exit_knowledge: bool,
) -> Scenario:
    # disable communication so initial exit knowledge remains the controlled variable
    return Scenario(
        building=create_demo_building(),
        agent_specs=create_demo_agents(
            full_exit_knowledge=full_exit_knowledge,
            communication_likelihood=0.0,
        ),
        hazard_cells=frozenset({
            (4, 4),
        }),
        alarm_time=0.0,
    )


def clear_terminal() -> None:
    os.system("cls" if os.name == "nt" else "clear")


def create_frame(
    simulation: Simulation,
    blocked_agent_ids: frozenset[str] = frozenset(),
) -> str:
    # build one display frame from the current simulation state
    scenario = simulation.scenario
    building = scenario.building

    display: list[list[str]] = []

    for y in range(building.height):
        row: list[str] = []

        for x in range(building.width):
            coordinate = (x, y)

            if coordinate in scenario.hazard_cells:
                symbol = "H"

            else:
                cell_type = building.get_cell(x, y)

                if cell_type == CellType.WALL:
                    symbol = "#"

                elif cell_type == CellType.EXIT:
                    symbol = "E"

                else:
                    symbol = "."

            row.append(symbol)

        display.append(row)

    # active agents are rendered over the underlying building cells
    for agent in simulation.agents:
        if agent.evacuated:
            continue

        x, y = agent.position

        if agent.id in blocked_agent_ids:
            display[y][x] = "X"

        else:
            display[y][x] = "A"

    evacuated = sum(
        agent.evacuated
        for agent in simulation.agents
    )

    lines = [
        "EgressSim Prototype",
        "",
        (
            f"simulation time: "
            f"{simulation.time:.1f}s / "
            f"{MAX_SIMULATION_TIME:.1f}s"
        ),
        f"timestep: {simulation.timestep:.1f}s",
        f"evacuated: {evacuated}/{len(simulation.agents)}",
        "",
    ]

    for row in display:
        lines.append(" ".join(row))

    lines.extend([
        "",
        "# wall   E exit   H hazard   A agent   X blocked",
    ])

    return "\n".join(lines)


def display_frame(frame: str) -> None:
    clear_terminal()
    print(frame)


def print_results(simulation: Simulation) -> None:
    result = simulation.get_result()

    print()
    print("Simulation Results")
    print("------------------")
    print(f"elapsed time: {result.elapsed_time:.1f}s")

    if result.total_evacuation_time is not None:
        print(
            f"total evacuation time: "
            f"{result.total_evacuation_time:.1f}s"
        )

    print(
        f"evacuated: "
        f"{result.evacuated_agents}/{result.total_agents}"
    )
    print(f"remaining: {result.remaining_agents}")

    print(
        f"blocked movement attempts: "
        f"{result.blocked_movement_attempts}"
    )

    print(f"termination: {result.termination_reason}")
    print(f"seed: {result.seed}")

    print()
    print("exit usage:")

    for exit_id, count in result.exit_usage.items():
        print(f"  {exit_id}: {count}")


def review_frames(frames: list[str]) -> None:
    # allow the completed simulation to be inspected one displayed tick at a time
    if not frames:
        return

    index = 0

    while True:
        display_frame(frames[index])

        print()
        print(f"frame {index + 1}/{len(frames)}")
        print("[p] previous   [n] next   [q] quit")

        command = input("> ").strip().lower()

        if command == "p":
            index = max(0, index - 1)

        elif command == "n":
            index = min(
                len(frames) - 1,
                index + 1,
            )

        elif command == "q":
            return


def format_exit_knowledge(
    exit_ids: set[str] | frozenset[str],
) -> str:
    if not exit_ids:
        return "none"

    return ", ".join(sorted(exit_ids))


def print_knowledge_comparison(
    full_scenario: Scenario,
    limited_scenario: Scenario,
) -> None:
    # show the exact initial exit knowledge used in each comparison run
    print("Initial Exit Knowledge")
    print("----------------------")
    print()

    print(
        f"{'Agent':<12}"
        f"{'Full Knowledge':<32}"
        f"{'Limited Knowledge':<32}"
    )

    print("-" * 76)

    full_agents = {
        agent.id: agent
        for agent in full_scenario.agent_specs
    }

    limited_agents = {
        agent.id: agent
        for agent in limited_scenario.agent_specs
    }

    for agent_id in full_agents:
        full_knowledge = format_exit_knowledge(
            full_agents[agent_id].known_exit_ids
        )

        limited_knowledge = format_exit_knowledge(
            limited_agents[agent_id].known_exit_ids
        )

        print(
            f"{agent_id:<12}"
            f"{full_knowledge:<32}"
            f"{limited_knowledge:<32}"
        )


def print_comparison_setup(
    full_scenario: Scenario,
    limited_scenario: Scenario,
) -> None:
    # display the shared setup for both the initial and previous screens
    print("EgressSim Exit Knowledge Comparison")
    print()

    print(
        "Limited exit knowledge means agents may know only a subset of "
        "the available exits."
    )

    print()

    print(
        "Both runs use the same building, agents, speeds, reaction times, "
        "and simulation seed."
    )

    print(
        "Communication is disabled so initial exit knowledge "
        "stays controlled."
    )

    print()

    print_knowledge_comparison(
        full_scenario=full_scenario,
        limited_scenario=limited_scenario,
    )


def run_visual_demo() -> None:
    scenario = create_demo_scenario()

    simulation = Simulation(
        scenario=scenario,
        seed=42,
        timestep=0.5,
    )

    frames: list[str] = []

    initial_frame = create_frame(simulation)
    frames.append(initial_frame)

    display_frame(initial_frame)

    print()
    print("Press Enter to start the simulation.\n")
    print("[q] quit")

    command = input("> ").strip().lower()

    if command == "q":
        return

    # step manually so each simulation state can be displayed and saved
    while (
        not simulation.all_agents_evacuated
        and simulation.time + simulation.timestep
        <= MAX_SIMULATION_TIME
    ):
        resolution = simulation.step()

        blocked_agent_ids = frozenset(
            proposal.agent_id
            for proposal in resolution.blocked
        )

        frame = create_frame(
            simulation=simulation,
            blocked_agent_ids=blocked_agent_ids,
        )

        frames.append(frame)

        display_frame(frame)
        time.sleep(DISPLAY_DELAY)

    if not simulation.all_agents_evacuated:
        # finalize a manually stepped run that reached the configured time limit
        simulation.run(
            max_time=MAX_SIMULATION_TIME,
        )

    print_results(simulation)

    print()
    review = input(
        "Review simulation ticks? [y/N] "
    ).strip().lower()

    if review == "y":
        review_frames(frames)

        # restore the final state and results after leaving review mode
        display_frame(frames[-1])
        print_results(simulation)

    print()
    input("Press Enter to return to the menu.")


def run_knowledge_comparison() -> None:
    # compare identical scenarios where only initial exit knowledge changes
    full_scenario = create_comparison_scenario(
        full_exit_knowledge=True,
    )

    limited_scenario = create_comparison_scenario(
        full_exit_knowledge=False,
    )

    full_simulation = Simulation(
        scenario=full_scenario,
        seed=42,
        timestep=0.5,
    )

    limited_simulation = Simulation(
        scenario=limited_scenario,
        seed=42,
        timestep=0.5,
    )

    # show the setup before running so the comparison can be inspected first
    clear_terminal()

    print_comparison_setup(
        full_scenario=full_scenario,
        limited_scenario=limited_scenario,
    )

    print()
    print("Press Enter to run the comparison.\n")
    print("[q] quit")

    command = input("> ").strip().lower()

    if command == "q":
        return

    full_simulation.run(
        max_time=MAX_SIMULATION_TIME,
    )

    limited_simulation.run(
        max_time=MAX_SIMULATION_TIME,
    )

    full_result = full_simulation.get_result()
    limited_result = limited_simulation.get_result()

    while True:
        clear_terminal()

        print("EgressSim Exit Knowledge Comparison")
        print()

        print(
            f"{'Metric':<30}"
            f"{'Full Knowledge':>18}"
            f"{'Limited Knowledge':>20}"
        )

        print("-" * 68)

        print(
            f"{'Evacuated':<30}"
            f"{full_result.evacuated_agents:>18}"
            f"{limited_result.evacuated_agents:>20}"
        )

        print(
            f"{'Remaining':<30}"
            f"{full_result.remaining_agents:>18}"
            f"{limited_result.remaining_agents:>20}"
        )

        full_time = (
            f"{full_result.total_evacuation_time:.1f}s"
            if full_result.total_evacuation_time is not None
            else "not completed"
        )

        limited_time = (
            f"{limited_result.total_evacuation_time:.1f}s"
            if limited_result.total_evacuation_time is not None
            else "not completed"
        )

        print(
            f"{'Total evacuation time':<30}"
            f"{full_time:>18}"
            f"{limited_time:>20}"
        )

        print(
            f"{'Blocked movement attempts':<30}"
            f"{full_result.blocked_movement_attempts:>18}"
            f"{limited_result.blocked_movement_attempts:>20}"
        )

        print(
            f"{'exit_right usage':<30}"
            f"{full_result.exit_usage['exit_right']:>18}"
            f"{limited_result.exit_usage['exit_right']:>20}"
        )

        print(
            f"{'exit_bottom usage':<30}"
            f"{full_result.exit_usage['exit_bottom']:>18}"
            f"{limited_result.exit_usage['exit_bottom']:>20}"
        )

        print()
        print("[p] previous   [q] quit")

        command = input("> ").strip().lower()

        if command == "q":
            return

        if command == "p":
            while True:
                clear_terminal()

                print_comparison_setup(
                    full_scenario=full_scenario,
                    limited_scenario=limited_scenario,
                )

                print()
                print("[n] results   [q] quit")

                command = input("> ").strip().lower()

                if command == "q":
                    return

                if command == "n":
                    break


def main() -> None:
    while True:
        clear_terminal()

        print("EgressSim Prototype")
        print()
        print("[1] Run visual evacuation demo")
        print("[2] Run exit-knowledge comparison")
        print("[q] Quit")
        print()

        command = input("> ").strip().lower()

        if command == "1":
            run_visual_demo()

        elif command == "2":
            run_knowledge_comparison()

        elif command == "q":
            return


if __name__ == "__main__":
    main()
