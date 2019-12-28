import torch
import torch.nn.functional as F
from torch_scatter import scatter_max
from torch_geometric.data import Batch

from data.constants import Transition

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def optimize_model(memory, policy_net, target_net, optimizer, batch_size, gamma):
    # Sample a batch from buffer if available
    if len(memory) < batch_size:
        return
    batch = memory.sample(batch_size)

    # Compute a mask of non-final states and concatenate the batch elements
    # in a single graph using pytorch_geometric Batch class
    non_final_mask = torch.tensor(
        tuple(map(lambda s: s is not None, batch.next_state)),
        device=device,
        dtype=torch.bool,
    )

    if any(non_final_mask):
        non_final_next_states = Batch.from_data_list(
            [s for s in batch.next_state if s is not None]
        ).to(device)

    state_batch = Batch.from_data_list(batch.state)
    # state_batch = state_batch.to(device)
    action_batch = torch.stack(batch.action)
    reward_batch = torch.stack(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net

    state_action_values = torch.zeros(batch_size, device=device)
    state_action_values_batch = policy_net(state_batch.x, state_batch.edge_index)

    # TODO Try to remove for loop
    for i in range(batch_size):
        state_action_values[i] = state_action_values_batch[state_batch.batch == i][
            action_batch[i]
        ]

    # Compute V(s_{t+1}) for all next states.
    with torch.no_grad():
        next_state_values = torch.zeros((batch_size, 1), device=device)
        target_prediction = target_net(
            non_final_next_states.x, non_final_next_states.edge_index
        )
        neighbor_mask = non_final_next_states.mask.squeeze()
        if any(non_final_mask):
            next_state_values[non_final_mask], _ = scatter_max(
                target_prediction[neighbor_mask],
                non_final_next_states.batch[neighbor_mask],
                dim=0,
            )
        next_state_values = next_state_values.squeeze()

    # Compute the expected Q values
    expected_state_action_values = (next_state_values * gamma) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

