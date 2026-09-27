# defines movement proposals and resolves conflicts between agents trying to move
from collections.abc import Collection
from dataclasses import dataclass
from random import Random

from .coordinate import Coordinate


@dataclass(frozen=True)
class MoveProposal:

    agent_id: str
    destination: Coordinate


@dataclass(frozen=True)
class MovementResolution:

    accepted: tuple[MoveProposal, ...]
    blocked: tuple[MoveProposal, ...]


def resolve_move_conflicts(
    proposals: tuple[MoveProposal, ...],
    occupied_cells: Collection[Coordinate],
    rng: Random,
) -> MovementResolution:
    # each agent can attempt at most one move during a movement phase
    agent_ids = [proposal.agent_id for proposal in proposals]

    if len(set(agent_ids)) != len(agent_ids):
        raise ValueError("Each agent may submit only one move proposal.")

    proposals_by_destination: dict[
        Coordinate,
        list[MoveProposal],
    ] = {}

    for proposal in proposals:
        proposals_by_destination.setdefault(
            proposal.destination,
            [],
        ).append(proposal)

    accepted: list[MoveProposal] = []
    blocked: list[MoveProposal] = []

    # sort destinations and agents so proposal ordering does not affect seeded results
    for destination, destination_proposals in sorted(
        proposals_by_destination.items()
    ):
        candidates = sorted(
            destination_proposals,
            key=lambda proposal: proposal.agent_id,
        )

        # cells occupied at the start of the movement phase cannot be entered
        if destination in occupied_cells:
            blocked.extend(candidates)
            continue

        if len(candidates) == 1:
            accepted.append(candidates[0])
            continue

        winner = rng.choice(candidates)

        accepted.append(winner)

        blocked.extend(
            proposal
            for proposal in candidates
            if proposal != winner
        )

    return MovementResolution(
        accepted=tuple(
            sorted(
                accepted,
                key=lambda proposal: proposal.agent_id,
            )
        ),
        blocked=tuple(
            sorted(
                blocked,
                key=lambda proposal: proposal.agent_id,
            )
        ),
    )