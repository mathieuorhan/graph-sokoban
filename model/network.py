import torch
import torch.nn as nn
import torch.nn.functional as F

from model.message_passing import GCNConv


class Net(torch.nn.Module):
    def __init__(self, nodes_features=4, num_message_passing=6):
        super(Net, self).__init__()
        self.conv_i = GCNConv(nodes_features, 64)
        self.convs_h = nn.Sequential(
            *[GCNConv(64, 64) for _ in range(num_message_passing)]
        )
        self.conv_o = GCNConv(64, 1)

    def forward(self, x, edge_index):
        # Apply graph layers
        x = self.conv_i(x, edge_index)
        x = F.relu(x)
        for conv in self.convs_h:
            x = conv(x, edge_index)
            x = F.relu(x)
        x = self.conv_o(x, edge_index)
        return x
