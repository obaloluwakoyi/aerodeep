"""
milestone2/fusion/node_fusion.py

Fuses time-series embeddings and text (maintenance log) embeddings
at the graph node level.

Strategy:
  - Time-series embedding: (timeseries_dim,) from Stream A encoder
  - Text embedding: (text_dim,) from Stream B BERT embedder
    → Retrieved from pgvector by nearest-neighbour lookup on
      (unit_id, node component mentions, recency)
  - Fusion: project both to a common dim, concatenate, apply
    a learned gating mechanism, then project to final fusion_dim

The gating is important: for nodes with few log mentions (e.g. intercooler
rarely has dedicated log entries) the gate should weight towards the
time-series signal. For nodes with rich log history (HP cylinder),
the text signal carries diagnostic context.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from loguru import logger


class GatedNodeFusion(nn.Module):
    """
    Gated multimodal fusion for a single graph node.

    Architecture:
      ts_proj:   (timeseries_dim) → (hidden_dim)
      txt_proj:  (text_dim) → (hidden_dim)
      gate:      (hidden_dim * 2) → scalar in (0,1)
      output:    (hidden_dim * 2) → (fusion_dim)

    The gate determines how much of the text signal to blend with the
    time-series signal. When text embedding is a zero vector (no log
    available), the gate naturally collapses to 0.
    """

    def __init__(
        self,
        timeseries_dim: int = 128,
        text_dim: int = 768,
        hidden_dim: int = 256,
        fusion_dim: int = 256,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.ts_proj = nn.Sequential(
            nn.Linear(timeseries_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        self.txt_proj = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # Gating network: takes both projected signals, outputs scalar gate
        self.gate_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

        # Final output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        ts_emb: torch.Tensor,      # (batch, timeseries_dim) or (timeseries_dim,)
        txt_emb: torch.Tensor,     # (batch, text_dim) or (text_dim,)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            fused:  (batch, fusion_dim) or (fusion_dim,)
            gate:   (batch, 1) or (1,) — text gate weight (0=ts only, 1=full text)
        """
        unbatched = ts_emb.dim() == 1
        if unbatched:
            ts_emb = ts_emb.unsqueeze(0)
            txt_emb = txt_emb.unsqueeze(0)

        ts_h = self.ts_proj(ts_emb)     # (B, hidden)
        txt_h = self.txt_proj(txt_emb)  # (B, hidden)

        # Gate: blend text in proportion to signal quality
        gate = self.gate_net(torch.cat([ts_h, txt_h], dim=-1))  # (B, 1)

        # Gated blend: ts always present, text modulated by gate
        blended = torch.cat([ts_h, gate * txt_h], dim=-1)       # (B, hidden*2)
        fused = self.output_proj(blended)                        # (B, fusion_dim)

        if unbatched:
            return fused.squeeze(0), gate.squeeze(0)
        return fused, gate


class MultiNodeFusionLayer(nn.Module):
    """
    Manages one GatedNodeFusion per graph node.

    In training: processes a batch of (node_features, text_embeddings)
    In inference: processes individual nodes as they arrive
    """

    def __init__(
        self,
        node_ids: List[str],
        timeseries_dim: int = 128,
        text_dim: int = 768,
        hidden_dim: int = 256,
        fusion_dim: int = 256,
        dropout: float = 0.2,
        share_weights: bool = False,
    ):
        super().__init__()
        self._node_ids = node_ids
        self._fusion_dim = fusion_dim

        if share_weights:
            # Single shared fusion module — parameter efficient
            shared = GatedNodeFusion(timeseries_dim, text_dim, hidden_dim, fusion_dim, dropout)
            self.fusers = nn.ModuleDict({nid: shared for nid in node_ids})
        else:
            # Independent fusion per node — more capacity, learns node-specific patterns
            self.fusers = nn.ModuleDict({
                nid: GatedNodeFusion(timeseries_dim, text_dim, hidden_dim, fusion_dim, dropout)
                for nid in node_ids
            })

    def forward(
        self,
        ts_embeddings: Dict[str, torch.Tensor],
        txt_embeddings: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            ts_embeddings:  {node_id: (timeseries_dim,) or (B, timeseries_dim)}
            txt_embeddings: {node_id: (text_dim,) or (B, text_dim)}
                            May contain zero vectors for nodes with no recent logs

        Returns:
            fused_matrix:  (N_nodes, fusion_dim) — ordered by node_ids
            gate_values:   {node_id: gate_scalar} — for dashboard display
        """
        fused_rows = []
        gate_values = {}

        for nid in self._node_ids:
            ts = ts_embeddings[nid]
            txt = txt_embeddings.get(nid, torch.zeros_like(ts).expand(-1, 768) if ts.dim() > 1 else torch.zeros(768))

            # Handle device
            device = ts.device
            if isinstance(txt, np.ndarray):
                txt = torch.from_numpy(txt).float().to(device)
            else:
                txt = txt.to(device)

            fused, gate = self.fusers[nid](ts, txt)
            fused_rows.append(fused)
            gate_values[nid] = gate

        return torch.stack(fused_rows, dim=0), gate_values

    @property
    def output_dim(self) -> int:
        return self._fusion_dim


class TextEmbeddingRetriever:
    """
    Retrieves the most relevant historical text embedding for each node
    from the vector store, given recent sensor anomaly context.

    This bridges Stream B (vector store) with the graph fusion layer.
    """

    def __init__(
        self,
        vector_store,           # VectorStore instance
        embedder,               # IndustrialEmbedder instance
        node_component_map: Dict[str, List[str]],
        embed_dim: int = 768,
        top_k: int = 3,
    ):
        self._vs = vector_store
        self._embedder = embedder
        self._node_component_map = node_component_map
        self._embed_dim = embed_dim
        self._top_k = top_k

    def retrieve_node_embeddings(
        self,
        unit_id: str,
        anomaly_context: Optional[str] = None,
    ) -> Dict[str, np.ndarray]:
        """
        For each node, retrieve the most relevant text embedding from
        the vector store. Falls back to zero vector if no relevant
        logs exist.

        Args:
            unit_id: compressor unit to search within
            anomaly_context: optional description of current anomaly
                             to bias retrieval (e.g. "high vibration HP bearing")
        Returns:
            {node_id: embedding (embed_dim,)}
        """
        node_embeddings = {}

        for node_id, components in self._node_component_map.items():
            # Build query from component names + anomaly context
            query_parts = [f"maintenance issues with {', '.join(components)}"]
            if anomaly_context:
                query_parts.append(anomaly_context)
            query = ". ".join(query_parts)

            query_emb = self._embedder.embed_query(query)
            results = self._vs.search_chunks(
                query_embedding=query_emb,
                top_k=self._top_k,
                unit_id=unit_id,
            )

            if results:
                # Average top-k chunk embeddings (weighted by similarity)
                # For simplicity, we re-encode the best chunk text
                best_texts = [r.chunk_text for r in results if r.chunk_text]
                if best_texts:
                    combined = " ".join(best_texts[:2])  # top 2 chunks
                    node_embeddings[node_id] = self._embedder.embed_query(combined)
                else:
                    node_embeddings[node_id] = np.zeros(self._embed_dim, dtype=np.float32)
            else:
                node_embeddings[node_id] = np.zeros(self._embed_dim, dtype=np.float32)

        return node_embeddings
