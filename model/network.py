import torch
import torch.nn.functional as F

from model.message_passing import GCNConv


class Net(torch.nn.Module):
    def __init__(self, nodes_features=4):
        super(Net, self).__init__()
        self.conv1 = GCNConv(nodes_features, 32)
        self.conv2 = GCNConv(32, 32)
        self.conv3 = GCNConv(32, 1)

    def forward(self, x, edge_index):
        # Apply graph layers
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        x = self.conv3(x, edge_index)
        return x
