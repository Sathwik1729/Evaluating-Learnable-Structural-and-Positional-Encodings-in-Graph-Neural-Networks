"""LSPE-GCN models for node classification.

The `LSPE` class is the upgraded, more paper-aligned variant:
- PE is projected to the hidden dimension before message passing.
- The positional stream uses a residual tanh update, preserving signed
  coordinates as in Dwivedi et al.
- Laplacian PE can be randomly sign-flipped during training.

`LegacyLSPE` preserves the original project implementation so experiments can
compare the old code against the improved model.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, GCNConv
from torch_geometric.utils import add_self_loops


class ImprovedLSPELayer(MessagePassing):
    """One lightweight LSPE layer with separate structural and positional streams."""

    def __init__(self, hidden_dim: int, pe_dim: int, dropout: float = 0.5):
        super().__init__(aggr='mean')
        self.dropout = dropout

        self.W_h = nn.Linear(hidden_dim + pe_dim, hidden_dim)
        self.W_p = nn.Linear(pe_dim, pe_dim)

        self.bn_h = nn.BatchNorm1d(hidden_dim)

    def forward(self, h: torch.Tensor, p: torch.Tensor,
                edge_index: torch.Tensor):
        edge_index_sl, _ = add_self_loops(edge_index, num_nodes=h.size(0))

        hp = torch.cat([h, p], dim=-1)
        hp_agg = self.propagate(edge_index_sl, x=hp)
        h_new = F.dropout(
            F.relu(self.bn_h(self.W_h(hp_agg))),
            p=self.dropout,
            training=self.training,
        )

        p_agg = self.propagate(edge_index_sl, x=p)
        p_new = p + torch.tanh(self.W_p(p_agg))

        return h_new, p_new

    def message(self, x_j: torch.Tensor) -> torch.Tensor:
        return x_j


class LegacyLSPELayer(MessagePassing):
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
    """Improved lightweight LSPE-GCN model.

    This remains a citation-graph adaptation, not a full GatedGCN/PNA
    reproduction. Compared with the legacy model, it keeps the parameter count
    close to the old code while using the paper's signed residual PE update.
    """

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int,
                 num_layers: int = 2, dropout: float = 0.5, pe_dim: int = 8,
                 lap_pe_sign_flip: bool = True, **kwargs):
        super().__init__()
        self.dropout = dropout
        self.pe_dim = pe_dim
        self.lap_pe_sign_flip = lap_pe_sign_flip

        self.input_conv = GCNConv(in_channels, hidden_channels)
        self.input_bn = nn.BatchNorm1d(hidden_channels)

        self.layers = nn.ModuleList([
            ImprovedLSPELayer(hidden_channels, pe_dim, dropout=dropout)
            for _ in range(num_layers)
        ])

        self.classifier = nn.Linear(hidden_channels, out_channels)

    def _initial_pe(self, data):
        if data.pe is None:
            raise ValueError("LSPE requires data.pe. Use a PE mode other than 'none'.")

        pe = data.pe
        pe_mode = getattr(data, 'pe_mode', None)
        if (
            self.training
            and self.lap_pe_sign_flip
            and pe_mode == 'laplacian'
        ):
            signs = torch.randint(
                0, 2, (pe.size(1),), device=pe.device, dtype=pe.dtype
            ).mul_(2).sub_(1)
            pe = pe * signs

        return pe

    def forward(self, data):
        h = F.relu(self.input_bn(self.input_conv(data.x, data.edge_index)))
        h = F.dropout(h, p=self.dropout, training=self.training)
        p = self._initial_pe(data)

        for layer in self.layers:
            h, p = layer(h, p, data.edge_index)

        return self.classifier(h)


class LegacyLSPE(nn.Module):
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
            LegacyLSPELayer(hidden_channels, pe_dim)
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
