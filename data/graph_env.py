import copy
import torch
from torch_geometric.data import Data

import data.utils as utils


class GraphEnv:
    """The goal of this class is to avoid using gym_sokoban and work
    directly on the graph. This enables to
    generate in advance levels, reuse them, etc.
    """

    STEP_REWARD = -0.1
    ON_BOX_REWARD = 1
    OFF_BOX_REWARD = -1
    FINISH_REWARD = 10

    def __init__(self, embedding, device, stop_if_unreachable=True):
        self.state = None
        self.embedding = embedding
        self.device = device
        self.stop_if_unreachable = stop_if_unreachable

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
        node_idx = node_idx.squeeze()
        assert (
            self.state is not None
        ), "Call reset with an initial state before using step"

        reward = self.STEP_REWARD

        if utils.all_boxes_on_all_targets(self.state):
            done = True
        else:
            done = False

        next_state = utils.clone_and_detach_data(self.state)

        # If self, do nothing
        if next_state.player_idx == node_idx:
            pass
        elif not node_idx.nelement():
            raise ValueError("node_idx is empty")
        elif not utils.is_neighbor_of_player(node_idx, next_state.mask):
            if self.stop_if_unreachable:
                raise ValueError("node_idx is unreachable")
            else:
                pass
        # If void, move
        elif not next_state.x[node_idx, 0] and not next_state.x[node_idx, 3]:
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
                if utils.all_boxes_on_all_targets(next_state):
                    reward += self.FINISH_REWARD
                next_state.player_idx = node_idx.long().unsqueeze(0)
            # Else, do nothing
            else:
                pass

        # Recompute mask
        next_state.mask = self.embedding.get_node_neighbors_mask(
            next_state.player_idx, next_state.edge_index, next_state.x
        ).to(self.device)

        info = {"deadlock": utils.are_off_target_boxes_in_corner(next_state)}
        self.state = next_state
        return next_state, reward, done, info

    def render(self):
        return self.state

