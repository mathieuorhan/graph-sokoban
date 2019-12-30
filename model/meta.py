import torch
import torch.nn as nn
from torch_scatter import scatter_mean
from torch_geometric.nn import MetaLayer


class EdgeModel(torch.nn.Module):
    def __init__(self, n_node_features, n_edge_features, hiddens, n_targets):
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * n_node_features + n_edge_features, hiddens),
            nn.ReLU(),
            nn.Linear(hiddens, n_targets),
        )

    def forward(self, src, dest, edge_attr, u=None, batch=None):
        out = torch.cat([src, dest, edge_attr], 1)
        out = self.edge_mlp(out)
        return out


class NodeModel(torch.nn.Module):
    def __init__(self, n_node_features, n_edge_features, hiddens, n_targets):
        super(NodeModel, self).__init__()
        self.node_mlp_1 = nn.Sequential(
            nn.Linear(n_node_features + n_edge_features, hiddens),
            nn.ReLU(),
            nn.Linear(hiddens, hiddens),
        )
        self.node_mlp_2 = nn.Sequential(
            nn.Linear(hiddens + n_node_features, hiddens),
            nn.ReLU(),
            nn.Linear(hiddens, n_targets),
        )

    def forward(self, x, edge_index, edge_attr, u, batch):
        row, col = edge_index
        # Note: edge attr are updated BEFORE nodes
        # Shape (n_edges, n_node_features + n_edges_features)
        out = torch.cat([x[col], edge_attr], dim=1)
        # Shape (n_edges, hiddens)
        out = self.node_mlp_1(out)
        # Shape (num_nodes, hiddens)
        out = scatter_mean(out, row, dim=0, dim_size=x.size(0))
        # Shape (num_nodes, n_node_features + hiddens)
        out = torch.cat([x, out], dim=1)
        # Shape (num_nodes, n_targets)
        out = self.node_mlp_2(out)
        return out


class MetaGNN(torch.nn.Module):
    def __init__(self, n_node_features, n_edge_features, hiddens=256):
        super().__init__()

        # First layer
        edge_model1 = EdgeModel(
            n_node_features=n_node_features,
            n_edge_features=n_edge_features,
            hiddens=hiddens,
            n_targets=hiddens,
        )
        node_model1 = NodeModel(
            n_node_features=n_node_features,
            n_edge_features=hiddens,
            hiddens=hiddens,
            n_targets=hiddens,
        )
        self.layer1 = MetaLayer(edge_model=edge_model1, node_model=node_model1)

        # Second layer
        edge_model2 = EdgeModel(
            n_node_features=hiddens,
            n_edge_features=hiddens,
            hiddens=hiddens,
            n_targets=hiddens,
        )
        node_model2 = NodeModel(
            n_node_features=hiddens,
            n_edge_features=hiddens,
            hiddens=hiddens,
            n_targets=1,
        )
        self.layer2 = MetaLayer(edge_model=edge_model2, node_model=node_model2)

    def forward(self, x, edge_index, edge_attr, u=None, batch=None):
        x, edge_attr, u = self.layer1(x, edge_index, edge_attr, u, batch)
        x, edge_attr, u = self.layer2(x, edge_index, edge_attr, u, batch)
        return x, edge_attr, u


if __name__ == "__main__":
    layer = MetaLayer(edge_model=EdgeModel(...), node_model=NodeModel(...),)

