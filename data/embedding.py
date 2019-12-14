import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.utils import remove_isolated_nodes

from constants import TinyWorldElements as elem


class MinimalEmbedding:
    NUM_NODES_FEATURES = 4
    RENDER_MODE = "tiny_rgb_array"

    @classmethod
    def __call__(cls, state_pixels):
        """Convert raw pixels to a graph

        Args:
            - state_pixels: (W, H, 3) np.uint8
            - embedding: function taking a (W, H, 3) and returning (W, H, N) Tensor
            N>= 4 and always include the embedding_minimal
            - linker: function taking a (W, H, 3) state and returning edges_index
        Return:
            - torch_geometric Data instance
        """
        # Pixel state
        state = torch.tensor(state_pixels, dtype=torch.uint8)

        # Nodes (original) coordinates (num_nodes, 2)
        num_nodes = state.size(0) * state.size(1)
        pos = np.stack(np.unravel_index(np.arange(num_nodes), state.shape[:2]), axis=1)

        # Nodes embedding (num_nodes, n_features)
        x = cls.embedding(state).float().reshape(num_nodes, -1)

        # Edges
        edges_index = cls.linker(state)

        # Remove isolated nodes
        edges_index, _, nodes_mask = remove_isolated_nodes(
            edges_index, num_nodes=num_nodes
        )
        x = x[nodes_mask]
        pos = torch.tensor(pos[nodes_mask], dtype=torch.long)

        # Action mask
        player_idx = cls.get_player_idx(x)  # (1,)
        mask = cls.get_node_neighbors_mask(player_idx, edges_index, x)
        mask = mask.unsqueeze(1).bool()  # (num_nodes, 1) bool

        # Apply mask
        graph = Data(
            x=x, pos=pos, edge_index=edges_index, mask=mask, player_idx=player_idx
        )

        return graph

    @classmethod
    def linker(cls, state):
        return cls.linker_neighbors(state)

    @classmethod
    def linker_neighbors(cls, state):
        """Link grid neighbors"""
        W, H = state.size(0), state.size(1)
        pos = torch.arange(W * H).reshape(W, H)
        right = torch.stack((pos[:, 1:], pos[:, :-1])).reshape(2, -1)
        left = right[[1, 0]].clone().detach()
        down = torch.stack((pos[:-1, :], pos[1:, :])).reshape(2, -1)
        up = down[[1, 0]].clone().detach()

        edges_index = torch.cat((up, down, left, right), axis=1)
        return edges_index

    @classmethod
    def linker_fc(cls, state):
        """Link every nodes (no self-edge)"""
        W, H = state.size(0), state.size(1)
        gridx, gridy = torch.meshgrid(torch.arange(W * H), torch.arange(W * H))
        gridx = gridx.reshape(-1)
        gridy = gridy.reshape(-1)
        no_self_edges_mask = gridx != gridy
        gridx = gridx[no_self_edges_mask]
        gridy = gridy[no_self_edges_mask]
        edges_index = torch.stack((gridx, gridy)).long().contiguous()
        return edges_index

    @classmethod
    def embedding(cls, state):
        """Minimal embedding of a pixel RGB state
        Return:
            (W, H, 4) tensor
        """

        has_box = torch.all(state == elem.BOX, -1) | torch.all(
            state == elem.BOX_ON_TARGET, -1
        )
        has_player = torch.all(state == elem.PLAYER, -1) | torch.all(
            state == elem.PLAYER_ON_TARGET, -1
        )
        is_target = (
            torch.all(state == elem.BOX_TARGET, -1)
            | torch.all(state == elem.PLAYER_ON_TARGET, -1)
            | torch.all(state == elem.BOX_ON_TARGET, -1)
        )
        is_wall = torch.all(state == elem.WALL, -1)

        embedded = torch.stack([has_box, has_player, is_target, is_wall], dim=-1)
        return embedded

    @classmethod
    def get_player_idx(cls, x):
        _player_idx = (x[:, 1] == 1).nonzero().item()
        player_idx = torch.tensor(_player_idx).long().unsqueeze(0)
        return player_idx

    @classmethod
    def get_node_neighbors_mask(cls, idx, edges_index, x):
        """Binary mask corresponding to nodes's neighbors (including itself)"""
        neighbors_index = edges_index[:, edges_index[0] == idx][1]
        mask = torch.zeros(x.size(0), dtype=torch.int32)
        mask[neighbors_index] = 1
        mask[idx] = 1
        return mask


class NoWallsEmbedding(MinimalEmbedding):
    """Similar to Minimal Embedding, but removes wall-wall links"""

    @classmethod
    def linker(cls, state):
        """Same as linker_neighbors but remove wall-wall links"""
        edges_index = cls.linker_neighbors(state)
        value2value = state.reshape(-1, 3)[edges_index]
        mask = torch.all(value2value[0] == elem.WALL, -1) & torch.all(
            value2value[1] == elem.WALL, -1
        )
        edges_index = edges_index[:, ~mask]
        return edges_index


if __name__ == "__main__":
    import gym
    import gym_sokoban

    render_mode = "tiny_rgb_array"  # One pixel for one state
    env = gym.make("Sokoban-small-v1")
    p = env.reset(render_mode=render_mode)

    E = MinimalEmbedding()

    G = E(p)