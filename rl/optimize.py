import torch
import torch.nn.functional as F

from data.constants import Transition

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def optimize_model(memory, policy_net, target_net, optimizer, batch_size, gamma):
    # Sample a batch from buffer if available
    if len(memory) >= batch_size:
        return
    batch = memory.sample(batch_size)

    raise NotImplementedError  # See https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
