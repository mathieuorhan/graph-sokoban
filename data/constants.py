import torch
from collections import namedtuple

Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))


class TinyWorldElements:
    WALL = torch.tensor([0, 0, 0], dtype=torch.uint8)
    FLOOR = torch.tensor([243, 248, 238], dtype=torch.uint8)
    BOX_TARGET = torch.tensor([254, 126, 125], dtype=torch.uint8)
    BOX_ON_TARGET = torch.tensor([254, 95, 56], dtype=torch.uint8)
    BOX = torch.tensor([142, 121, 56], dtype=torch.uint8)
    PLAYER = torch.tensor([160, 212, 56], dtype=torch.uint8)
    PLAYER_ON_TARGET = torch.tensor([219, 212, 56], dtype=torch.uint8)


# Map the difference between the chosen action node and the action node type (dx, dy, action_node_type) 
# to the action taken by the environment (int between 0 and 8)
NODES_TO_ACTIONS = {
    (0, 0, 1): 0,  # No operation
    (0, 0, 2): 0,  # No operation
    (1, 0, 0): 8,  # Move right
    (1, 0, 2): 4,  # Push right
    (1, 0, 3): 4,  # Push right
    (-1, 0, 0): 7,  # Move left
    (-1, 0, 2): 3,  # Push left
    (-1, 0, 3): 3,  # Push left
    (0, 1, 0): 6,  # Move down
    (0, 1, 2): 2,  # Push down
    (0, 1, 3): 2,  # Push down
    (0, -1, 0): 5,  # Move up
    (0, -1, 2): 1,  # Push up
    (0, -1, 3): 1,  # Push up
}

