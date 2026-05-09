"""GAT with positional encodings concatenated to input features (PEGAT).

Not a reproduction of Dwivedi et al. — a stronger project extension that feeds
PE directly into the GAT backbone and tests whether PE helps attention-based GNNs.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class PEGAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 num_layers=2, dropout=0.5, heads=4, pe_dim=8, **kwargs):
        super().__init__()
        self.dropout = dropout
        self.heads = heads

        assert hidden_channels % heads == 0, (
            f"hidden_channels ({hidden_channels}) must be divisible by heads ({heads})"
        )
        head_dim = hidden_channels // heads
        first_in = in_channels + pe_dim

        self.convs = nn.ModuleList()
        if num_layers == 1:
            self.convs.append(
                GATConv(first_in, out_channels, heads=1, concat=False, dropout=dropout)
            )
        else:
            self.convs.append(
                GATConv(first_in, head_dim, heads=heads, concat=True, dropout=dropout)
            )
            for _ in range(num_layers - 2):
                self.convs.append(
                    GATConv(hidden_channels, head_dim, heads=heads, concat=True, dropout=dropout)
                )
            self.convs.append(
                GATConv(hidden_channels, out_channels, heads=1, concat=False, dropout=dropout)
            )

    def forward(self, data):
        if data.pe is None:
            raise ValueError("PEGAT requires data.pe. Use a PE mode other than 'none'.")
        x = torch.cat([data.x, data.pe], dim=-1)
        x = F.dropout(x, p=self.dropout, training=self.training)
        for conv in self.convs[:-1]:
            x = conv(x, data.edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.convs[-1](x, data.edge_index)
