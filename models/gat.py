"""
TODO (Person 2): Implement GAT baseline.

Use torch_geometric.nn.GATConv.
Interface is identical to GCN — forward(self, data) returns logits [N, out_channels].

GATConv signature: GATConv(in_channels, out_channels, heads=heads, concat=True/False)
- For intermediate layers: concat=True, out_channels = hidden_channels // heads
- For final layer: concat=False, heads=1, out_channels = out_channels

The 'heads' hyperparam comes through **kwargs from the config (key: 'heads', default 4).
"""
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class GAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 num_layers=2, dropout=0.5, heads=4, **kwargs):
        super().__init__()
        self.dropout = dropout

        # TODO: build self.convs using GATConv
        # Note: when concat=True, each GATConv layer outputs heads * out_channels features.
        # So intermediate layers: GATConv(in, hidden // heads, heads=heads, concat=True)
        #                         → output size = hidden_channels
        # Final layer:            GATConv(hidden, out_channels, heads=1, concat=False)
        raise NotImplementedError("Person 2: build GAT layers in __init__")

    def forward(self, data):
        # TODO: implement — apply convs with ELU activation and dropout between layers
        # Use F.elu (not F.relu) — standard for GAT
        raise NotImplementedError("Person 2: implement GAT.forward()")
