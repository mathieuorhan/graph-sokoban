import torch
import torch.nn as nn
import torch.nn.functional as F

from model.message_passing import GCNConv, GATConv
from torch_geometric.nn import EdgeConv


class Net(torch.nn.Module):
    def __init__(self, nodes_features=4, num_message_passing=6, hiddens=64):
        super().__init__()
        self.conv_i = GCNConv(nodes_features, hiddens)
        self.convs_h = nn.Sequential(
            *[GCNConv(hiddens, hiddens) for _ in range(num_message_passing)]
        )
        self.conv_o = GCNConv(hiddens, 1)

    def forward(self, x, edge_index, batch=None):
        # Apply graph layers
        x = self.conv_i(x, edge_index)
        x = F.relu(x)
        for conv in self.convs_h:
            x = conv(x, edge_index)
            x = F.relu(x)
        x = self.conv_o(x, edge_index)
        return x


class SharedEdgeNet(torch.nn.Module):
    def __init__(self, nodes_features=4, num_core=8, hiddens=64):
        super().__init__()
        self.num_core = num_core
        self.encoder = EdgeConv(
            nn.Sequential(
                nn.Linear(2 * nodes_features, hiddens),
                nn.ReLU(),
                nn.Linear(hiddens, hiddens),
            )
        )
        self.core = EdgeConv(
            nn.Sequential(
                nn.Linear(2 * hiddens, hiddens), nn.ReLU(), nn.Linear(hiddens, hiddens),
            )
        )
        self.decoder = EdgeConv(
            nn.Sequential(
                nn.Linear(2 * hiddens, hiddens), nn.ReLU(), nn.Linear(hiddens, 1),
            )
        )

    def forward(self, x, edge_index, edge_attr, u=None, batch=None):
        # Apply graph layers
        x = self.encoder(x, edge_index)
        for _ in range(self.num_core):
            x = self.core(x, edge_index)
        x = self.decoder(x, edge_index)
        return x, edge_attr, u


class EdgeNet(torch.nn.Module):
    def __init__(self, nodes_features=4, num_message_passing=6, hiddens=64):
        super().__init__()
        self.conv_i = EdgeConv(
            nn.Sequential(
                nn.Linear(2 * nodes_features, hiddens),
                nn.ReLU(),
                nn.Linear(hiddens, hiddens),
            )
        )
        self.convs_h = nn.Sequential(
            *[
                EdgeConv(
                    nn.Sequential(
                        nn.Linear(2 * hiddens, hiddens),
                        nn.ReLU(),
                        nn.Linear(hiddens, hiddens),
                    )
                )
                for _ in range(num_message_passing)
            ]
        )
        self.conv_o = EdgeConv(
            nn.Sequential(
                nn.Linear(2 * hiddens, hiddens), nn.ReLU(), nn.Linear(hiddens, 1),
            )
        )

    def forward(self, x, edge_index, edge_attr, u=None, batch=None):
        # Apply graph layers
        x = self.conv_i(x, edge_index)
        x = F.relu(x)
        for conv in self.convs_h:
            x = conv(x, edge_index)
            x = F.relu(x)
        x = self.conv_o(x, edge_index)
        return x, edge_attr, u


class GATNet(torch.nn.Module):
    def __init__(self, nodes_features, num_message_passing=6, heads=1):
        super(GATNet, self).__init__()
        self.hidden = num_message_passing
        self.conv_i = GATConv(nodes_features, 64, heads=heads)
        self.lin_i = torch.nn.Linear(nodes_features, heads * 64)
        self.convs_h = torch.nn.Sequential(
            *[GATConv(heads * 64, 64, heads=heads) for _ in range(num_message_passing)]
        )
        self.lins_h = torch.nn.Sequential(
            *[
                torch.nn.Linear(heads * 64, heads * 64)
                for _ in range(num_message_passing)
            ]
        )
        self.conv_o = GATConv(heads * 64, 1, heads=heads, concat=False)
        self.lin_o = torch.nn.Linear(heads * 64, 1)

    def forward(self, x, edge_index, batch=None):
        x = F.elu(self.conv_i(x, edge_index) + self.lin_i(x))
        for i in range(self.hidden):
            x = F.elu(self.convs_h[i](x, edge_index) + self.lins_h[i](x))
        x = self.conv_o(x, edge_index) + self.lin_o(x)
        return x
