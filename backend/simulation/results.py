# defines the final metrics produced by a completed simulation run
from dataclasses import dataclass


@dataclass(frozen=True)
class SimulationResult:

    total_agents: int
    evacuated_agents: int
    remaining_agents: int
    elapsed_time: float
    total_evacuation_time: float | None
    exit_usage: dict[str, int]
    blocked_movement_attempts: int
    termination_reason: str
    seed: int