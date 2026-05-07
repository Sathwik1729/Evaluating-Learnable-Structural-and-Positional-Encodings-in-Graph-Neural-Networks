"""
GAT (Graph Attention Network) baseline model.

Uses torch_geometric.nn.GATConv.
Interface is identical to GCN — forward(self, data) returns logits [N, out_channels].

Key dimension accounting for multi-head attention:
  - Intermediate layers: concat=True, each head outputs (hidden_channels // heads) features
    → total output = heads * (hidden_channels // heads) = hidden_channels  ✓
  - Final layer: concat=False, heads=1 (average over heads), outputs out_channels
  - Activation: F.elu (standard for GAT per Veličković et al. 2018)
"""
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class GAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 num_layers=2, dropout=0.5, heads=4, **kwargs):
        super().__init__()
        self.dropout = dropout
        self.heads = heads

        assert hidden_channels % heads == 0, (
            f"hidden_channels ({hidden_channels}) must be divisible by heads ({heads})"
        )
        head_dim = hidden_channels // heads  # per-head output dim for intermediate layers

        self.convs = nn.ModuleList()

        if num_layers == 1:
            # Single layer: go directly to out_channels
            self.convs.append(
                GATConv(in_channels, out_channels, heads=1, concat=False,
                        dropout=dropout)
            )
        else:
            # First layer: in_channels → hidden_channels (via heads)
            self.convs.append(
                GATConv(in_channels, head_dim, heads=heads, concat=True,
                        dropout=dropout)
            )
            # Intermediate layers: hidden_channels → hidden_channels
            for _ in range(num_layers - 2):
                self.convs.append(
                    GATConv(hidden_channels, head_dim, heads=heads, concat=True,
                            dropout=dropout)
                )
            # Final layer: hidden_channels → out_channels (average over 1 head)
            self.convs.append(
                GATConv(hidden_channels, out_channels, heads=1, concat=False,
                        dropout=dropout)
            )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for conv in self.convs[:-1]:
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = conv(x, edge_index)
            x = F.elu(x)
        # Final layer — no activation, returns raw logits
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x
