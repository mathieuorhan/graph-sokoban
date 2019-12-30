import random
import torch


def best_from_nodes(scores, state):
    """Find the best node according to prediction among neighbors.

    Args:
        - scores: (num_nodes, 1) float Tensor
        - graph: input Data graph (unbatched)
    Return:
        - (tensor, long, ()) node indice corresponding to the best action
    """
    # Select the position and the type of the node corresponding to the max of the scores
    _, idx_best = torch.max(scores[state.mask], 0)
    idx_best = state.mask.nonzero()[idx_best][0]

    return idx_best


def epsilon_greedy(state, net, eps):
    """Epsilon-greedy exploration
    
    Args:
        - state: input Data graph (unbatched)
        - net: policy net
        - eps: epsilon
    Return:
        - (tensor, long, ()) node indice corresponding to the action  
    """
    if random.random() > eps:
        with torch.no_grad():
            scores = net(state.x, state.edge_index)
            chosen_idx = best_from_nodes(scores, state)
    else:
        # Sample randomly a node among the neighboring nodes of the player
        player_neighbors = state.mask.nonzero()
        chosen_idx = random.choice(player_neighbors)[0]
    return chosen_idx

