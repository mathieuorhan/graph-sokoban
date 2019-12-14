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

