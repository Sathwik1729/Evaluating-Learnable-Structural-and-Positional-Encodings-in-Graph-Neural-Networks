"""
TODO (Person 2): Implement GraphSAGE baseline.

Use torch_geometric.nn.SAGEConv.
Interface is identical to GCN — forward(self, data) returns logits [N, out_channels].
Copy the GCN structure and swap GCNConv → SAGEConv.
SAGEConv(in_channels, out_channels) has the same signature.
"""
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class GraphSAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 num_layers=2, dropout=0.5, **kwargs):
        super().__init__()
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

    def forward(self, data):
        # TODO: implement — same pattern as GCN
        raise NotImplementedError("Person 2: implement GraphSAGE.forward()")
