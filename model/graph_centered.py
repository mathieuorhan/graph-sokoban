import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    TopKPooling,
    SAGPooling,
    EdgeConv,
    global_mean_pool,
    global_max_pool,
    GCNConv,
)


class GraphCenteredNet(torch.nn.Module):
    """Custom graph to approach the problem as graph classification
    with 4 possibles outputs corresponding to the 4 directions.

    Apply several convolutions / pooling, and a global pooling, and a final MLP

    - Edge features are ignored for now
    - Position should be used in embedding to take adv. of EdgeConvs
    """

    def __init__(
        self,
        n_node_features,
        n_edge_features=None,
        hiddens=32,
        aggr="max",
        flow="source_to_target",
        ratio=1.0,
        device="cuda",
    ):
        super().__init__()
        assert hiddens % 4 == 0, "`hiddens` has to be a multiple of 4"
        self.device = device
        self.aggr = aggr
        self.flow = flow
        self.ratio = ratio

        # Build encoder
        self.conv_e, self.pool_e = self.build_block(
            n_node_features, hiddens, hiddens, ratio
        )

        # Build core
        self.conv_c1, self.pool_c1 = self.build_block(hiddens, hiddens, hiddens, ratio)
        self.conv_c2, self.pool_c2 = self.build_block(hiddens, hiddens, hiddens, ratio)

        # Build decoder
        self.decoder = nn.Sequential(
            nn.Linear(hiddens, hiddens // 2),
            nn.ReLU(),
            nn.Linear(hiddens // 2, hiddens // 4),
            nn.ReLU(),
            nn.Linear(hiddens // 4, 4),
        )

    def build_block(self, in_channels, out_channels, hiddens, ratio=1.0):
        mlp = nn.Sequential(
            nn.Linear(2 * in_channels, hiddens),
            nn.ReLU(),
            nn.Linear(hiddens, out_channels),
        )
        conv = EdgeConv(nn=mlp, aggr=self.aggr)
        if self.ratio < 1.0:
            pool = SAGPooling(out_channels, ratio=ratio)
        else:
            pool = None
        return conv, pool

    def forward(self, x, edge_index, edge_attr, u=None, batch=None):
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long).to(self.device)

        # Encoder
        x = F.relu(self.conv_e(x, edge_index))
        if self.ratio < 1:
            x, edge_index, _, batch, _, _ = self.pool_e(x, edge_index, None, batch)

        # Core
        x = F.relu(self.conv_c1(x, edge_index))
        if self.ratio < 1:
            x, edge_index, _, batch, _, _ = self.pool_c1(x, edge_index, None, batch)

        x = F.relu(self.conv_c2(x, edge_index))
        if self.ratio < 1:
            x, edge_index, _, batch, _, _ = self.pool_c2(x, edge_index, None, batch)

        # Global pooling
        z = global_max_pool(x, batch)

        # (batch_size, 4)
        probs = self.decoder(z).view(-1, 4)
        return probs, edge_attr, u


class GraphCenteredNetV2(torch.nn.Module):
    """Custom graph to approach the problem as graph classification
    with 4 possibles outputs corresponding to the 4 directions.

    Apply several convolutions / pooling, and a global pooling, and a final MLP

    - Edge features are ignored for now
    - Position should be used in embedding to take adv. of EdgeConvs
    """

    def __init__(
        self,
        n_node_features,
        n_edge_features=None,
        hiddens=32,
        num_message_passing=2,
        aggr="max",
        flow="source_to_target",
        device="cuda",
    ):
        super().__init__()
        assert hiddens % 4 == 0, "`hiddens` has to be a multiple of 4"
        self.device = device
        self.aggr = aggr
        self.flow = flow
        self.num_message_passing = num_message_passing

        # Build encoder
        self.conv_e = self.build_block(n_node_features, hiddens, hiddens)

        # Build core
        self.conv_c = nn.Sequential(
            *[
                self.build_block(hiddens, hiddens, hiddens)
                for _ in range(num_message_passing)
            ]
        )

        # Build decoder
        self.decoder = nn.Sequential(
            nn.Linear(hiddens, hiddens // 2),
            nn.ReLU(),
            nn.Linear(hiddens // 2, hiddens // 4),
            nn.ReLU(),
            nn.Linear(hiddens // 4, 4),
        )

    def build_block(self, in_channels, out_channels, hiddens):
        mlp = nn.Sequential(
            nn.Linear(2 * in_channels, hiddens),
            nn.ReLU(),
            nn.Linear(hiddens, out_channels),
        )
        conv = EdgeConv(nn=mlp, aggr=self.aggr)
        return conv

    def forward(self, x, edge_index, edge_attr, u=None, batch=None):
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long).to(self.device)

        # Encoder
        x = F.relu(self.conv_e(x, edge_index))

        # Core
        for i in range(self.num_message_passing):
            x = F.relu(self.conv_c[i](x, edge_index))

        # Global pooling
        z = global_max_pool(x, batch)

        # (batch_size, 4)
        probs = self.decoder(z).view(-1, 4)
        return probs, edge_attr, u


class SimpleGraphCenteredNet(torch.nn.Module):
    def __init__(
        self,
        n_node_features,
        n_edge_features=None,
        hiddens=32,
        aggr="max",
        depth=4,
        **kwargs
    ):
        super(SimpleGraphCenteredNet, self).__init__()
        assert hiddens % 4 == 0, "`hiddens` has to be a multiple of 4"
        self.hiddens = hiddens
        self.aggr = aggr
        self.depth = depth

        self.conv_i = GCNConv(n_node_features, hiddens)
        self.convs_h = nn.Sequential(*[GCNConv(hiddens, hiddens) for _ in range(depth)])

        self.decoder = nn.Sequential(
            nn.Linear(hiddens, hiddens // 2), nn.ReLU(), nn.Linear(hiddens // 2, 4),
        )

    def forward(self, x, edge_index, edge_attr, u=None, batch=None):
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long).to(self.device)

        x = F.relu(self.conv_i(x, edge_index))
        for conv in self.convs_h:
            x = F.relu(conv(x, edge_index))

        # Global pooling
        z = global_max_pool(x, batch)

        # (batch_size, 4)
        probs = self.decoder(z).view(-1, 4)
        return probs, edge_attr, u
