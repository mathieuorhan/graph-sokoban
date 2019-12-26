import random
import torch
from data.constants import NODES_TO_ACTIONS

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def best_from_nodes(scores, state):
    """
    Args:
        - scores: (num_nodes, 1) float Tensor
        - graph: Data original
    Return:
        - node indice corresponding to the best action
    """
    # Select the position and the type of the node corresponding to the max of the scores
    mask = state.mask
    _, idx_best = torch.max(scores[mask], 0)
    idx_best = mask.nonzero()[idx_best][0]

    return idx_best


def epsilon_greedy(state, net, eps, random_generator=None):
    """Epsilon-greedy exploration
    
    Args:
        - state: Data
        - net: policy net
        - eps: epsilon
        - random_generator
    Return:
        - node indice corresponding to the action
        - int in [0, ..., 8], action taken 
    """
    if random_generator is None:
        sample = random.random()
    else:
        sample = random_generator.random()
    if sample > eps:
        with torch.no_grad():
            scores = net(state.x.to(device), state.edge_index.to(device))
            chosen_idx = best_from_nodes(scores, state)

    else:
        # Sample randomly a node among the neighboring nodes of the player
        player_neighbors = state.edge_index[:, state.edge_index[0] == state.player_idx][
            1
        ]
        nb_neighbors = player_neighbors.size(0)
        action = random.randint(0, nb_neighbors + 1)
        if action == 0:
            chosen_idx = state.player_idx[0]
        else:
            chosen_idx = random.choice(player_neighbors)

    action_node_pos = state.pos[chosen_idx]
    # Whether or not there is a box on the chosen node
    action_node_box = 1 if state.x[chosen_idx][0] == 1 else 0

    # 9 possibilities corresponding to the 9 possible actions
    diff_pos = action_node_pos - state.pos[state.player_idx]
    dy, dx = diff_pos[0, 0].item(), diff_pos[0, 1].item()

    return (
        torch.tensor([[chosen_idx]], dtype=torch.long, device=device),
        NODES_TO_ACTIONS[(dx, dy, action_node_box)],
    )

