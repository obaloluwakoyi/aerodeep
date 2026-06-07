"""
milestone3/model/dual_head.py

Full AeroDeep diagnostic model: wraps the fusion layer + ST-GCN encoder
into a single end-to-end nn.Module for training and inference.

Combines:
  - MultiNodeFusionLayer  (milestone2/fusion/node_fusion.py)
  - AeroDeepSTGCN         (milestone2/fusion/stgcn.py)
into one trainable unit with:
  - Head 1: TTF regression  — hours to next failure event
  - Head 2: Multi-label fault classification — (N_fault_classes,)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from milestone2.graph.schema import CompressorGraphSchema
from milestone2.fusion.node_fusion import MultiNodeFusionLayer
from milestone2.fusion.stgcn import AeroDeepSTGCN


@dataclass
class DiagnosticOutput:
    """Structured inference output for the API and dashboard."""
    unit_id: str
    window_ms: int

    # TTF prediction
    ttf_hours: float
    ttf_confidence: float           # 1 - normalised prediction uncertainty

    # Fault predictions
    fault_probabilities: Dict[str, float]   # {fault_label: probability}
    top_faults: List[Tuple[str, float]]     # sorted by probability desc

    # Node-level risk (for graph visualisation)
    node_risk: Dict[str, float]             # {node_id: risk_score}

    # Text gate values (how much log context contributed per node)
    text_gate_values: Dict[str, float]


class AeroDeepDiagnosticModel(nn.Module):
    """
    End-to-end AeroDeep diagnostic model.

    Forward pass:
      1. Fuse time-series and text embeddings at each node
      2. Build T-length graph sequence
      3. ST-GCN encoder + dual heads
      4. Return TTF + fault logits + node risk
    """

    def __init__(
        self,
        timeseries_dim: int = 128,
        text_dim: int = 768,
        fusion_dim: int = 256,
        hidden_channels: List[int] = None,
        temporal_kernel_size: int = 9,
        n_fault_classes: int = 18,
        dropout: float = 0.3,
        share_fusion_weights: bool = False,
    ):
        super().__init__()

        if hidden_channels is None:
            hidden_channels = [256, 128, 64]

        node_ids = CompressorGraphSchema.node_ids()

        self.fusion_layer = MultiNodeFusionLayer(
            node_ids=node_ids,
            timeseries_dim=timeseries_dim,
            text_dim=text_dim,
            hidden_dim=fusion_dim,
            fusion_dim=fusion_dim,
            dropout=dropout,
            share_weights=share_fusion_weights,
        )

        self.stgcn = AeroDeepSTGCN(
            node_feature_dim=fusion_dim,
            n_fault_classes=n_fault_classes,
            hidden_channels=hidden_channels,
            temporal_kernel_size=temporal_kernel_size,
            edge_dim=2,
            heads=4,
            dropout=dropout,
        )

        self._n_nodes = len(node_ids)
        self._node_ids = node_ids
        self._n_faults = n_fault_classes

    def forward(
        self,
        ts_sequence: Dict[str, torch.Tensor],   # {node_id: (T, ts_dim)}
        txt_embeddings: Dict[str, torch.Tensor],# {node_id: (text_dim,)}
        edge_index: torch.Tensor,               # (2, E)
        edge_attr: torch.Tensor,                # (E, 2)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """
        Returns:
            ttf_pred:     (1,)
            fault_logits: (N_faults,)
            node_risk:    (N_nodes,)
            gate_values:  {node_id: tensor}
        """
        T = next(iter(ts_sequence.values())).shape[0]

        # Fuse at each timestep
        x_fused_sequence = []
        gate_values = None

        for t in range(T):
            ts_t = {nid: ts_sequence[nid][t] for nid in self._node_ids}
            fused_t, gate_t = self.fusion_layer(ts_t, txt_embeddings)
            x_fused_sequence.append(fused_t)
            if t == T - 1:
                gate_values = gate_t

        x_fused = torch.stack(x_fused_sequence, dim=0)  # (T, N, fusion_dim)

        ttf_pred, fault_logits, node_risk = self.stgcn(
            x_fused, edge_index, edge_attr
        )

        return ttf_pred, fault_logits, node_risk, gate_values

    @torch.no_grad()
    def predict(
        self,
        ts_sequence: Dict[str, np.ndarray],
        txt_embeddings: Dict[str, np.ndarray],
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        fault_threshold: float = 0.45,
        unit_id: str = "",
        window_ms: int = 0,
    ) -> DiagnosticOutput:
        """
        Inference-mode prediction. Accepts numpy arrays, returns DiagnosticOutput.
        """
        self.eval()
        device = next(self.parameters()).device

        # Convert to tensors
        ts_tensors = {
            nid: torch.from_numpy(arr).float().to(device)
            for nid, arr in ts_sequence.items()
        }
        txt_tensors = {
            nid: torch.from_numpy(arr).float().to(device)
            for nid, arr in txt_embeddings.items()
        }

        ttf_pred, fault_logits, node_risk, gate_values = self.forward(
            ts_tensors, txt_tensors,
            edge_index.to(device), edge_attr.to(device)
        )

        # TTF
        ttf_hours = float(ttf_pred.item())
        # Simple confidence: higher TTF = lower urgency = higher confidence
        ttf_confidence = float(torch.sigmoid(torch.tensor(ttf_hours / 168.0)))  # 168h = 1 week

        # Fault probabilities
        fault_probs = torch.sigmoid(fault_logits).cpu().numpy()
        fault_prob_dict = {
            CompressorGraphSchema.fault_label_by_id(i): float(fault_probs[i])
            for i in range(self._n_faults)
        }
        top_faults = sorted(fault_prob_dict.items(), key=lambda x: x[1], reverse=True)[:5]

        # Node risk
        node_risk_dict = {
            self._node_ids[i]: float(node_risk[i].item())
            for i in range(self._n_nodes)
        }

        # Gate values
        gate_dict = {
            nid: float(gate_values[nid].item())
            for nid in self._node_ids
            if nid in gate_values
        }

        return DiagnosticOutput(
            unit_id=unit_id,
            window_ms=window_ms,
            ttf_hours=max(0.0, ttf_hours),
            ttf_confidence=ttf_confidence,
            fault_probabilities=fault_prob_dict,
            top_faults=top_faults,
            node_risk=node_risk_dict,
            text_gate_values=gate_dict,
        )
