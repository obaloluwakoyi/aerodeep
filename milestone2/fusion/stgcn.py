"""
milestone2/fusion/stgcn.py

Spatio-Temporal Graph Convolutional Network (ST-GCN) for compressor
fault propagation modelling.

Architecture:
  Input: sequence of T graph snapshots, each with shape (N, F)
  Spatial:  GCN layers model how fault signatures propagate across
            physical relationships (edges) at each timestep
  Temporal: 1D convolution along the time axis models how signatures
            evolve over the T-window (typically 30-60 graph snapshots
            = 15-30 minutes at 30-second window intervals)
  Output:   Graph-level readout → dual heads (TTF + fault classification)
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATv2Conv, global_mean_pool, global_max_pool
from torch_geometric.data import Data, Batch
from loguru import logger


class SpatialGCNBlock(nn.Module):
    """
    One spatial block: GATv2 attention convolution + residual + LayerNorm.
    GATv2 is preferred over plain GCN because it learns *asymmetric*
    attention weights — important for directed relationships like
    fluid_flow (LP→intercooler→HP) vs lubrication (oil→bearings).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        edge_dim: int = 2,
        heads: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()
        assert out_channels % heads == 0, "out_channels must be divisible by heads"
        self.conv = GATv2Conv(
            in_channels,
            out_channels // heads,
            heads=heads,
            edge_dim=edge_dim,
            dropout=dropout,
            concat=True,
        )
        self.norm = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout(dropout)

        # Residual projection if dims change
        self.residual_proj = (
            nn.Linear(in_channels, out_channels)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        residual = self.residual_proj(x)
        out = self.conv(x, edge_index, edge_attr)
        out = self.dropout(out)
        return self.norm(out + residual)


class TemporalConvBlock(nn.Module):
    """
    Temporal convolution over T timesteps for each node independently.
    Uses dilated 1D conv to capture patterns at multiple timescales.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 9,
        dilation: int = 1,
        dropout: float = 0.2,
    ):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2  # 'same' padding
        self.conv = nn.Conv1d(
            channels, channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.norm = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (N_nodes * batch, channels, T)
        """
        out = self.conv(x)
        out = self.norm(out)
        out = F.gelu(out)
        return self.dropout(out) + x  # residual


class STGCNLayer(nn.Module):
    """
    One ST-GCN layer: spatial → temporal.
    Processes a T-length sequence of graph snapshots.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temporal_kernel_size: int = 9,
        edge_dim: int = 2,
        heads: int = 4,
        dropout: float = 0.2,
        temporal_dilation: int = 1,
    ):
        super().__init__()
        self.spatial = SpatialGCNBlock(
            in_channels, out_channels, edge_dim, heads, dropout
        )
        self.temporal = TemporalConvBlock(
            out_channels, temporal_kernel_size, temporal_dilation, dropout
        )

    def forward(
        self,
        x: torch.Tensor,          # (T, N, in_channels)
        edge_index: torch.Tensor,  # (2, E)
        edge_attr: torch.Tensor,   # (E, edge_dim)
    ) -> torch.Tensor:
        T, N, C = x.shape

        # ── Spatial pass: process each timestep independently ─────────────────
        x_spatial = []
        for t in range(T):
            out = self.spatial(x[t], edge_index, edge_attr)  # (N, out_channels)
            x_spatial.append(out)
        x_spatial = torch.stack(x_spatial, dim=0)  # (T, N, out_channels)

        # ── Temporal pass: process each node's time series ────────────────────
        # Reshape: (N, out_channels, T) for Conv1d
        x_t = x_spatial.permute(1, 2, 0)     # (N, out_channels, T)
        x_t = self.temporal(x_t)              # (N, out_channels, T)
        x_t = x_t.permute(2, 0, 1)           # (T, N, out_channels)

        return x_t


class STGCNEncoder(nn.Module):
    """
    Stack of ST-GCN layers with increasing dilation for multi-scale
    temporal modelling.

    Input:  sequence of T graph snapshots
    Output: (T, N, hidden_channels[-1]) node embeddings across time
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: List[int] = (256, 128, 64),
        temporal_kernel_size: int = 9,
        edge_dim: int = 2,
        heads: int = 4,
        dropout: float = 0.3,
    ):
        super().__init__()

        layers = []
        ch_in = in_channels
        dilations = [1, 2, 4]  # multi-scale temporal receptive field
        for i, ch_out in enumerate(hidden_channels):
            dilation = dilations[i] if i < len(dilations) else 4
            layers.append(STGCNLayer(
                ch_in, ch_out,
                temporal_kernel_size=temporal_kernel_size,
                edge_dim=edge_dim,
                heads=heads,
                dropout=dropout,
                temporal_dilation=dilation,
            ))
            ch_in = ch_out

        self.layers = nn.ModuleList(layers)
        self.out_channels = hidden_channels[-1]

    def forward(
        self,
        x: torch.Tensor,           # (T, N, in_channels)
        edge_index: torch.Tensor,  # (2, E)
        edge_attr: torch.Tensor,   # (E, edge_dim)
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, edge_index, edge_attr)
        return x                   # (T, N, out_channels)


class GraphReadout(nn.Module):
    """
    Aggregates node embeddings from the final timestep into a
    single graph-level representation.

    Strategy:
      - Take the last T timestep (most recent information)
      - Mean-pool + max-pool across nodes
      - Concatenate → (2 * out_channels)
    """

    def __init__(self, node_channels: int):
        super().__init__()
        self.out_dim = node_channels * 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (T, N, node_channels)
        Returns: (node_channels * 2,) graph representation
        """
        x_last = x[-1]                          # (N, node_channels) — last timestep
        mean_pool = x_last.mean(dim=0)           # (node_channels,)
        max_pool = x_last.max(dim=0).values      # (node_channels,)
        return torch.cat([mean_pool, max_pool])  # (node_channels * 2,)


class AeroDeepSTGCN(nn.Module):
    """
    Full AeroDeep ST-GCN model with dual output heads.

    Input:  T-length sequence of compressor graph snapshots
    Output:
      - ttf_pred:    (1,) predicted hours to failure
      - fault_logits: (N_fault_classes,) raw logits for multi-label classification
      - node_risk:    (N,) per-node risk scores for dashboard visualisation
    """

    def __init__(
        self,
        node_feature_dim: int,
        n_fault_classes: int,
        hidden_channels: List[int] = (256, 128, 64),
        temporal_kernel_size: int = 9,
        edge_dim: int = 2,
        heads: int = 4,
        dropout: float = 0.3,
        ttf_hidden: List[int] = (64, 32),
        fault_hidden: List[int] = (64, 32),
    ):
        super().__init__()

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(node_feature_dim, hidden_channels[0]),
            nn.LayerNorm(hidden_channels[0]),
            nn.GELU(),
        )

        # ST-GCN encoder
        self.encoder = STGCNEncoder(
            in_channels=hidden_channels[0],
            hidden_channels=hidden_channels,
            temporal_kernel_size=temporal_kernel_size,
            edge_dim=edge_dim,
            heads=heads,
            dropout=dropout,
        )

        # Node risk scoring (before readout — for dashboard)
        self.node_risk_head = nn.Sequential(
            nn.Linear(hidden_channels[-1], 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

        # Graph readout
        self.readout = GraphReadout(hidden_channels[-1])
        graph_repr_dim = hidden_channels[-1] * 2

        # TTF regression head
        self.ttf_head = self._make_head(graph_repr_dim, ttf_hidden, 1)

        # Fault classification head
        self.fault_head = self._make_head(graph_repr_dim, fault_hidden, n_fault_classes)

    def forward(
        self,
        x_sequence: torch.Tensor,   # (T, N, node_feature_dim)
        edge_index: torch.Tensor,    # (2, E)
        edge_attr: torch.Tensor,     # (E, edge_dim)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            ttf_pred:     (1,) — hours to failure
            fault_logits: (N_fault_classes,) — pre-sigmoid fault scores
            node_risk:    (N,) — per-node risk [0,1] for dashboard
        """
        T, N, F = x_sequence.shape

        # Project input features
        x = x_sequence.view(T * N, F)
        x = self.input_proj(x)
        x = x.view(T, N, -1)

        # ST-GCN forward
        x_enc = self.encoder(x, edge_index, edge_attr)  # (T, N, hidden[-1])

        # Node-level risk scores from last timestep
        node_risk = self.node_risk_head(x_enc[-1]).squeeze(-1)  # (N,)

        # Graph readout
        graph_repr = self.readout(x_enc)  # (hidden[-1] * 2,)

        # Dual heads
        ttf_pred = self.ttf_head(graph_repr)           # (1,)
        fault_logits = self.fault_head(graph_repr)     # (N_fault_classes,)

        return ttf_pred, fault_logits, node_risk

    @staticmethod
    def _make_head(in_dim: int, hidden_dims: List[int], out_dim: int) -> nn.Sequential:
        layers = []
        cur_dim = in_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(cur_dim, h), nn.GELU(), nn.Dropout(0.2)])
            cur_dim = h
        layers.append(nn.Linear(cur_dim, out_dim))
        return nn.Sequential(*layers)
