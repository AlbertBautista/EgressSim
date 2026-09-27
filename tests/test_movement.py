# tests movement conflict rules independently from the simulation engine
import random

import pytest

from backend.simulation.movement import (
    MoveProposal,
    resolve_move_conflicts,
)


# verifies agents can move when they request different unoccupied cells
def test_resolver_accepts_moves_to_distinct_empty_cells():
    proposals = (
        MoveProposal("agent_1", (1, 0)),
        MoveProposal("agent_2", (3, 0)),
    )

    resolution = resolve_move_conflicts(
        proposals=proposals,
        occupied_cells={(0, 0), (2, 0)},
        rng=random.Random(42),
    )

    assert resolution.accepted == proposals
    assert resolution.blocked == ()


# verifies a cell occupied at the start of movement cannot be entered
@pytest.mark.parametrize(
    ("proposals", "accepted_ids", "blocked_ids"),
    [
        pytest.param(
            (MoveProposal("agent_1", (1, 0)),),
            (),
            ("agent_1",),
            id="stationary-occupant",
        ),
        pytest.param(
            (
                MoveProposal("agent_1", (1, 0)),
                MoveProposal("agent_2", (0, 0)),
            ),
            (),
            ("agent_1", "agent_2"),
            id="swap",
        ),
        pytest.param(
            (
                MoveProposal("agent_1", (1, 0)),
                MoveProposal("agent_2", (2, 0)),
            ),
            ("agent_2",),
            ("agent_1",),
            id="chain-only-leading-agent-moves",
        ),
    ],
)
def test_resolver_blocks_move_into_occupied_cell(proposals, accepted_ids, blocked_ids):
    # agent_1 starts at (0, 0) and agent_2 starts at (1, 0)

    resolution = resolve_move_conflicts(
        proposals=proposals,
        occupied_cells={(0, 0), (1, 0)},
        rng=random.Random(42),
    )

    assert tuple(proposal.agent_id for proposal in resolution.accepted) == accepted_ids
    assert tuple(proposal.agent_id for proposal in resolution.blocked) == blocked_ids


# verifies seeded winners for multiple contested cells are independent of proposal order
def test_resolver_uses_seeded_randomness_for_conflicts():
    proposals = (
        MoveProposal("agent_1", (1, 1)),
        MoveProposal("agent_2", (1, 1)),
        MoveProposal("agent_3", (3, 1)),
        MoveProposal("agent_4", (3, 1)),
    )

    reordered_proposals = (
        # reverse candidates within each destination
        (proposals[1], proposals[0], proposals[3], proposals[2]),
        # reverse the destination groups without reversing their candidates
        (proposals[2], proposals[3], proposals[0], proposals[1]),
        tuple(reversed(proposals)),
    )

    occupied_cells = {(0, 1), (1, 0), (3, 0), (4, 1)}
    winning_proposals = set()

    # fixed seeds exercise different winners without requiring any particular winner per seed
    for seed in range(10):
        resolution = resolve_move_conflicts(
            proposals=proposals,
            occupied_cells=occupied_cells,
            rng=random.Random(seed),
        )

        assert len(resolution.accepted) == 2
        assert len(resolution.blocked) == 2
        assert {proposal.destination for proposal in resolution.accepted} == {
            (1, 1),
            (3, 1),
        }

        # every submitted proposal must appear exactly once, either accepted or blocked
        assert set(resolution.accepted).isdisjoint(resolution.blocked)
        assert set(resolution.accepted + resolution.blocked) == set(proposals)

        for proposal_order in reordered_proposals:
            reordered_resolution = resolve_move_conflicts(
                proposals=proposal_order,
                occupied_cells=occupied_cells,
                rng=random.Random(seed),
            )

            assert reordered_resolution == resolution

        winning_proposals.update(resolution.accepted)

    # neither destination should always favor the same agent regardless of the seed
    assert winning_proposals == set(proposals)


# verifies one agent cannot attempt multiple destinations in the same phase
def test_resolver_rejects_multiple_proposals_from_same_agent():
    proposals = (
        MoveProposal("agent_1", (1, 0)),
        MoveProposal("agent_1", (0, 1)),
    )

    with pytest.raises(ValueError):
        resolve_move_conflicts(
            proposals=proposals,
            occupied_cells={(0, 0)},
            rng=random.Random(42),
        )
