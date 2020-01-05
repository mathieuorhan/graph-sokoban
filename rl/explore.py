import random
import torch

import data.utils as utils


def epsilon_greedy_gc(state, policy_net, eps, device, opt):
    """Epsilon-greedy exploration
    
    Args:
        - state: input Data graph (unbatched)
        - net: policy net
        - eps: epsilon
    Return:
        - (tensor, long, (1,)) direction corresponding to the action 
    """
    if random.random() > eps:
        with torch.no_grad():
            scores, _, _ = policy_net(
                x=state.x,
                edge_index=state.edge_index,
                edge_attr=state.edge_attr,
                u=None,
                batch=None,
            )
            chosen_dir = torch.argmax(scores, dim=-1)
    else:
        if opt.sensible_moves_gc:
            nb_edge_idxs = (state.edge_index[0, :] == state.player_idx).nonzero().squeeze()
            sensible_moves = state.edge_attr[nb_edge_idxs].nonzero()[:, 0].squeeze()
            move_idx = torch.randint(sensible_moves.size(0), (1,), device=device)
            chosen_dir = sensible_moves[move_idx]
        else:
            # All moves
            chosen_dir = torch.randint(4, (1,), device=device)  
    return chosen_dir


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


def epsilon_greedy(state, policy_net, eps):
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
            scores, _, _ = policy_net(
                x=state.x,
                edge_index=state.edge_index,
                edge_attr=state.edge_attr,
                u=None,
                batch=None,
            )
            chosen_idx = best_from_nodes(scores, state)
    else:
        # Sample randomly a node among the neighboring nodes of the player
        player_neighbors = state.mask.nonzero()
        chosen_idx = random.choice(player_neighbors)[0]
    return chosen_idx

