import copy
import torch
from torch_geometric.data import Data


def clone_and_detach_data(data):
    return Data.from_dict(
        {
            k: v.clone().detach() if torch.is_tensor(v) else copy.deepcopy(v)
            for k, v in data.__dict__.items()
        }
    )


class GraphEnv:
    """The goal of this class is to avoid using gym_sokoban and work
    directly on the graph. This enables to
    generate in advance levels, reuse them, etc.
    """

    STEP_REWARD = -0.1
    ON_BOX_REWARD = 1
    OFF_BOX_REWARD = -1
    FINISH_REWARD = 10

    def __init__(self):
        self.state = None

    def reset(self, init_state):
        """
        Args:
            - init_state: Data, the graph
        """
        self.state = init_state

    def step(self, node_idx):
        """Gym-style step, except one provides a node index instead of an action.

        Args:
            - node_idx
        Returns:
            - new_state: Data, the graph of the new state
            - reward: float
            - done: bool
            - info: dict
        """
        assert (
            self.state is not None
        ), "Call reset with an initial state before using step"

        reward = self.STEP_REWARD

        if all(self.state.x[self.state.x[:, 2] == 1][:, 0]):
            done = True
        else:
            done = False

        next_state = clone_and_detach_data(self.state)

        # If self, do nothing
        if next_state.player_idx == node_idx:
            pass
        else:
            assert self.is_neighbor(node_idx), "node_idx is not reachable"
        # If void, move
        if not next_state.x[node_idx, 0] and not next_state.x[node_idx, 3]:
            next_state.x[next_state.player_idx, 1] = 0
            next_state.x[node_idx, 1] = 1
            next_state.player_idx = node_idx.long().unsqueeze(0)
        # If wall, do nothing
        elif next_state.x[node_idx, 3]:
            pass
        # If box, check next case behind
        elif next_state.x[node_idx, 0]:
            # Compute case coordinates
            diff_pos = next_state.pos[node_idx] - next_state.pos[next_state.player_idx]
            # print(next_state)
            # print(diff_pos)
            # dy, dx = diff_pos.squeeze()[0].item(), diff_pos.squeeze()[1].item()
            behind_pos = next_state.pos[node_idx] + diff_pos.squeeze()
            behind_idx = (
                torch.all(torch.eq(next_state.pos, behind_pos), dim=-1).nonzero().item()
            )

            # If next case is void, move the box and the player
            if not next_state.x[behind_idx, 0] and not next_state.x[behind_idx, 3]:
                next_state.x[next_state.player_idx, 1] = 0
                next_state.x[node_idx, 1] = 1
                next_state.x[node_idx, 0] = 0
                next_state.x[behind_idx, 0] = 1

                # Reward : if we move the box to a target
                if next_state.x[behind_idx, 2] == 1:
                    reward += self.ON_BOX_REWARD
                # Reward : if we move the box off a target
                if next_state.x[node_idx, 2] == 1:
                    reward += self.OFF_BOX_REWARD
                # Reward : if all boxes are on all targets
                if all(next_state.x[next_state.x[:, 2] == 1][:, 0]):
                    reward += self.FINISH_REWARD
                next_state.player_idx = node_idx.long().unsqueeze(0)
            # Else, do nothing
            else:
                pass

        # Recompute mask
        neighbors_index = next_state.edge_index[
            :, next_state.edge_index[0] == next_state.player_idx
        ][1]
        mask = torch.zeros_like(next_state.mask)
        mask[neighbors_index] = 1
        mask[next_state.player_idx] = 1
        next_state.mask = mask

        info = dict()
        self.state = next_state
        return next_state, reward, done, info

    def is_neighbor(self, node_idx):
        player_neighbors = self.state.edge_index[
            :, self.state.edge_index[0] == self.state.player_idx
        ][1]
        return node_idx in player_neighbors

    def render(self):
        return self.state
