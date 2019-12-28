import copy

import torch
from torch_geometric.data import Data


def all_boxes_on_all_targets(state):
    return all(state.x[state.x[:, 2] == 1][:, 0])


def is_neighbor_of_player(state, node_idx):
    """Check if a node is in the neighborhood of the player"""
    player_neighbors = state.edge_index[:, state.edge_index[0] == state.player_idx][1]
    return node_idx in player_neighbors


def clone_and_detach_data(state):
    """Clone and detach a torch_gemetric.data.Data"""
    return Data.from_dict(
        {
            k: v.clone().detach() if torch.is_tensor(v) else copy.deepcopy(v)
            for k, v in state.__dict__.items()
        }
    )


def are_off_target_boxes_in_corner(state):
    """Verify if any off target boxes is stuck in a corner

    This is useful to stop the game early and/or penalize
    """
    # Gather boxes not on targets
    boxes_idx = ((state.x[:, 0] == 1) & (state.x[:, 2] == 0)).nonzero().view(-1)
    nb_boxes = boxes_idx.size(0)

    # Count number of walls in their neighbors
    nb_walls = 0
    for box_idx in boxes_idx:
        neighbors_idx = state.edge_index[:, state.edge_index[0] == box_idx][1]
        nb_walls += (state.x[neighbors_idx, 3] == 1).sum().item()

    # Theshold
    is_deadlock = nb_walls > nb_boxes

    return is_deadlock


def count_boxes(state):
    return (state.x[:, 0] == 1).sum().item()

