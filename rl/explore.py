import random
import torch
from data.constants import NODES_TO_ACTIONS

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def epsilon_greedy(state, net, eps):
    """Epsilon-greedy exploration
    
    Args:
        - state: Data
        - net: policy net
        - eps: epsilon
    Return:
        - int in [0, ..., 8], action taken 
    """
    sample = random.random()
    if sample > eps:
        with torch.no_grad():
            scores = net(state.x, state.edge_index)
            action = best_from_nodes(scores, state)
            return action
    else:
        return torch.tensor([[random.randrange(8)]], device=device, dtype=torch.long)


def best_from_nodes(scores, state):
    """
    Args:
        - scores: (num_nodes, 1) float Tensor
        - graph: Data original
    Return:
        - int, action of node with best Q value
    """
    # Select the position and the type of the node corresponding to the max of the scores
    mask = state.mask
    _, idx_best = torch.max(scores[mask], 0)
    idx_best = mask.nonzero()[idx_best][0]
    action_node_pos = state.pos[idx_best]
    action_node_type = state.x[idx_best].nonzero().item()

    # 9 possibilities corresponding to the 9 possible actions
    diff_pos = action_node_pos - state.pos[state.player_idx]
    dx, dy = diff_pos[0, 0].item(), diff_pos[0, 1].item()

    return torch.tensor([[NODES_TO_ACTIONS[(dx, dy, action_node_type)]]])