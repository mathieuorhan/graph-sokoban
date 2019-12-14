import random
import torch

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
        return torch.tensor([[random.randrange(2)]], device=device, dtype=torch.long)


def best_from_nodes(scores, state):
    """
    Args:
        - scores: (num_nodes, 1) float Tensor
        - graph: Data original
    Return:
        - int, action of node with best Q value
    """
    raise NotImplementedError
