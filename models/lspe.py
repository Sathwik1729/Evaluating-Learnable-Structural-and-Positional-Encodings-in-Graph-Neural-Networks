"""
TODO (Person 2): Implement LSPE-GCN.

Paper: Dwivedi et al., "Graph Neural Networks with Learnable Structural and
Positional Representations", NeurIPS 2022. https://arxiv.org/abs/2110.07875

ARCHITECTURE (Section 3.2 of paper):
======================================
Input:
  h_v^0 = linear projection of node features x_v    [N, hidden_dim]
  p_v^0 = Laplacian eigenvectors (data.pe)           [N, pe_dim]

Each LSPE layer l:
  h_v^{l+1} = ReLU( W_h · MEAN_{u ∈ N(v)∪{v}}( concat(h_u^l, p_u^l) ) )
  p_v^{l+1} = ReLU( W_p · MEAN_{u ∈ N(v)∪{v}}( p_u^l ) )

  W_h has shape (hidden_dim + pe_dim) → hidden_dim
  W_p has shape pe_dim → pe_dim

Final:
  logits = linear( h_v^L )    [N, out_channels]
  (p is NOT used for classification, only for message passing)

KEY INSIGHT: p gets its OWN GNN update path, not just concatenated at input.
This lets the model refine positional information across layers.

IMPLEMENTATION STEPS:
1. Build LSPELayer (MessagePassing with aggr='mean', uses add_self_loops)
2. Build LSPE model stacking LSPELayer + classifier head
3. forward() reads data.x and data.pe
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree


class LSPELayer(MessagePassing):
    """
    One LSPE-GCN layer. Updates both h and p streams.

    Args:
        h_dim: feature dimension (consistent across layers)
        pe_dim: positional encoding dimension (consistent across layers)
    """
    def __init__(self, h_dim, pe_dim):
        super().__init__(aggr='mean')
        # W_h takes concat(h, p) and outputs h
        self.W_h = nn.Linear(h_dim + pe_dim, h_dim)
        # W_p takes p and outputs p (same dim)
        self.W_p = nn.Linear(pe_dim, pe_dim)

    def forward(self, h, p, edge_index):
        """
        Args:
            h: [N, h_dim] node features
            p: [N, pe_dim] positional encodings
            edge_index: [2, E]
        Returns:
            h_new: [N, h_dim]
            p_new: [N, pe_dim]
        """
        # TODO:
        # 1. Add self-loops to edge_index so each node aggregates itself too
        #    edge_index, _ = add_self_loops(edge_index, num_nodes=h.size(0))
        #
        # 2. Propagate for h update:
        #    hp = torch.cat([h, p], dim=-1)   # [N, h_dim + pe_dim]
        #    h_agg = self.propagate(edge_index, x=hp)  # [N, h_dim + pe_dim]
        #    h_new = F.relu(self.W_h(h_agg))
        #
        # 3. Propagate for p update (separate, p only):
        #    p_agg = self.propagate(edge_index, x=p)   # [N, pe_dim]
        #    p_new = F.relu(self.W_p(p_agg))
        #
        # 4. Return h_new, p_new
        raise NotImplementedError("Person 2: implement LSPELayer.forward()")

    def message(self, x_j):
        return x_j


class LSPE(nn.Module):
    """
    LSPE-GCN model for node classification.

    Args:
        in_channels: raw node feature dimension (from dataset)
        hidden_channels: hidden/feature dimension used in all layers
        out_channels: number of classes
        num_layers: number of LSPE layers
        dropout: dropout rate applied to h between layers
        pe_dim: positional encoding dimension (must match dataset loader pe_dim)
    """
    def __init__(self, in_channels, hidden_channels, out_channels,
                 num_layers=2, dropout=0.5, pe_dim=8, **kwargs):
        super().__init__()
        self.dropout = dropout
        self.pe_dim = pe_dim

        # TODO:
        # 1. Input projection: map raw features to hidden_dim
        #    self.input_proj = nn.Linear(in_channels, hidden_channels)
        #
        # 2. Stack of LSPELayer
        #    self.layers = nn.ModuleList([
        #        LSPELayer(hidden_channels, pe_dim) for _ in range(num_layers)
        #    ])
        #
        # 3. Classifier head
        #    self.classifier = nn.Linear(hidden_channels, out_channels)
        raise NotImplementedError("Person 2: build LSPE layers in __init__")

    def forward(self, data):
        """
        data.x:  [N, in_channels]   raw node features
        data.pe: [N, pe_dim]        Laplacian eigenvector PE
        Returns: logits [N, out_channels]
        """
        # TODO:
        # h = F.relu(self.input_proj(data.x))
        # p = data.pe
        # for layer in self.layers:
        #     h, p = layer(h, p, data.edge_index)
        #     h = F.dropout(h, p=self.dropout, training=self.training)
        # return self.classifier(h)
        raise NotImplementedError("Person 2: implement LSPE.forward()")
