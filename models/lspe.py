"""
LSPE-GCN: Graph Neural Network with Learnable Structural and Positional Encodings.

Paper: Dwivedi et al., "Graph Neural Networks with Learnable Structural and
Positional Representations", NeurIPS 2022. https://arxiv.org/abs/2110.07875

ARCHITECTURE:
======================================
Input encoding (h^0):
  h_v^0 = ReLU( BN( GCNConv(x_v, N(v)) ) )   [N, hidden_dim]
  p_v^0 = data.pe                               [N, pe_dim]

  NOTE: We use GCNConv (not a plain Linear) for the initial encoding so that
  the expensive feature dimension reduction (e.g. 3703→128 on Citeseer)
  already benefits from graph-structure aggregation. This matches the spirit
  of GNN-based encoding used in the original paper.

Each LSPE layer l (Section 3.2 of paper):
  h_v^{l+1} = ReLU( BN( W_h · MEAN_{u ∈ N(v)∪{v}}( concat(h_u^l, p_u^l) ) ) )
  p_v^{l+1} = ReLU( BN( W_p · MEAN_{u ∈ N(v)∪{v}}( p_u^l ) ) )

  W_h: (hidden_dim + pe_dim) → hidden_dim
  W_p: pe_dim → pe_dim

Final:
  logits = W_cls · h_v^L                 [N, out_channels]
  (p is NOT used for classification — only for message-passing enrichment)

KEY INSIGHTS:
  1. p gets its OWN GNN update path alongside h, not just concatenated at input.
     This lets the model refine positional information across all layers.
  2. Self-loops are added inside each LSPELayer so MEAN is over N(v) ∪ {v}.
  3. BatchNorm after each projection stabilises training on small labelled sets.
  4. GCNConv for input encoding provides graph-aware compression (critical for
     high-dimensional datasets like Citeseer with 3703 features).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, GCNConv
from torch_geometric.utils import add_self_loops


class LSPELayer(MessagePassing):
    """
    One LSPE-GCN layer. Updates both h (feature) and p (positional) streams.

    Two independent mean-aggregation passes per layer:
      1. h update: mean-aggregate concat(h, p) over N(v)∪{v} → project → BN → ReLU
      2. p update: mean-aggregate p over N(v)∪{v} → project → BN → ReLU

    Args:
        h_dim  (int): feature dimension, consistent across all layers
        pe_dim (int): positional encoding dimension, consistent across all layers
    """

    def __init__(self, h_dim: int, pe_dim: int):
        super().__init__(aggr='mean')
        # W_h: (h_dim + pe_dim) → h_dim
        self.W_h = nn.Linear(h_dim + pe_dim, h_dim, bias=True)
        # W_p: pe_dim → pe_dim  (same dim in/out — refines PE each layer)
        self.W_p = nn.Linear(pe_dim, pe_dim, bias=True)
        # BatchNorm stabilises training on small label sets
        self.bn_h = nn.BatchNorm1d(h_dim)
        self.bn_p = nn.BatchNorm1d(pe_dim)

    def forward(self, h: torch.Tensor, p: torch.Tensor,
                edge_index: torch.Tensor):
        """
        Args:
            h          : [N, h_dim]   node feature vectors
            p          : [N, pe_dim]  positional encoding vectors
            edge_index : [2, E]       graph connectivity
        Returns:
            h_new : [N, h_dim]
            p_new : [N, pe_dim]
        """
        # Add self-loops: MEAN_{u ∈ N(v) ∪ {v}}(·)  as per the paper
        edge_index_sl, _ = add_self_loops(edge_index, num_nodes=h.size(0))

        # ── h update ──────────────────────────────────────────────────────────
        hp = torch.cat([h, p], dim=-1)                # [N, h_dim + pe_dim]
        hp_agg = self.propagate(edge_index_sl, x=hp)  # [N, h_dim + pe_dim]
        h_new = F.relu(self.bn_h(self.W_h(hp_agg)))  # [N, h_dim]

        # ── p update (independent: aggregates p only, not h) ─────────────────
        p_agg = self.propagate(edge_index_sl, x=p)    # [N, pe_dim]
        p_new = F.relu(self.bn_p(self.W_p(p_agg)))   # [N, pe_dim]

        return h_new, p_new

    def message(self, x_j: torch.Tensor) -> torch.Tensor:
        return x_j


class LSPE(nn.Module):
    """
    LSPE-GCN model for node classification.

    Two parallel streams:
      h : node feature embeddings (used for final classification)
      p : positional encodings   (drives richer h updates, refined separately)

    Args:
        in_channels     (int)   raw input feature dim (from dataset)
        hidden_channels (int)   hidden dim used throughout all LSPE layers
        out_channels    (int)   number of classes
        num_layers      (int)   number of LSPE message-passing layers
        dropout         (float) dropout applied to h between layers
        pe_dim          (int)   PE dim; must match pe_dim in load_dataset()
    """

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int,
                 num_layers: int = 2, dropout: float = 0.5, pe_dim: int = 8,
                 **kwargs):
        super().__init__()
        self.dropout = dropout
        self.pe_dim = pe_dim

        # ── Input encoding: GCNConv for graph-aware feature compression ───────
        # Using GCNConv (not plain Linear) means the first layer already
        # aggregates neighbour features, critical when in_channels is large
        # (e.g. Citeseer: 3703→128 compression benefits from neighbour support).
        self.input_conv = GCNConv(in_channels, hidden_channels)
        self.input_bn = nn.BatchNorm1d(hidden_channels)

        # ── LSPE layers ───────────────────────────────────────────────────────
        self.layers = nn.ModuleList([
            LSPELayer(hidden_channels, pe_dim)
            for _ in range(num_layers)
        ])

        # ── Classifier — uses h only, NOT p ───────────────────────────────────
        self.classifier = nn.Linear(hidden_channels, out_channels)

    def forward(self, data):
        """
        Args:
            data.x         : [N, in_channels]  raw node features
            data.pe        : [N, pe_dim]        Laplacian eigenvector PE
            data.edge_index: [2, E]

        Returns:
            logits : [N, out_channels]
        """
        # ── Graph-aware initial encoding ──────────────────────────────────────
        h = F.relu(self.input_bn(self.input_conv(data.x, data.edge_index)))
        h = F.dropout(h, p=self.dropout, training=self.training)
        p = data.pe                           # [N, pe_dim]

        # ── LSPE message-passing layers ───────────────────────────────────────
        for layer in self.layers:
            h, p = layer(h, p, data.edge_index)
            h = F.dropout(h, p=self.dropout, training=self.training)

        # ── Classify using h only ─────────────────────────────────────────────
        return self.classifier(h)
