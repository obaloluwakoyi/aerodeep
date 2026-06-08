"""
milestone2/graph/builder.py

Constructs PyTorch Geometric (PyG) Data objects from fused node feature
vectors. Handles both static graph construction (for batch training) and
dynamic graph updates (for live inference).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch_geometric.data import Data, Batch
from loguru import logger

from milestone2.graph.schema import CompressorGraphSchema


class CompressorGraphBuilder:
    """
    Builds PyTorch Geometric Data objects from per-node fused embeddings.

    Each Data object represents the compressor graph at a single timestep
    window with:
      - x:           (N_nodes, node_feature_dim) fused node features
      - edge_index:  (2, N_edges) directed edges
      - edge_attr:   (N_edges, 2) [weight, relation_type_encoded]
      - y_ttf:       (1,) optional time-to-failure label in hours
      - y_fault:     (N_fault_classes,) optional multi-hot fault labels
      - window_ms:   scalar window timestamp
      - unit_id:     string unit identifier
    """

    RELATION_TYPE_MAP = {
        "fluid_flow": 0,
        "mechanical_drive": 1,
        "lubrication": 2,
        "sealing": 3,
        "thermal_proximity": 4,
        "structural_proximity": 5,
    }

    def __init__(self, node_feature_dim: int):
        self._schema = CompressorGraphSchema
        self._node_feature_dim = node_feature_dim
        self._n_nodes = len(self._schema.NODES)
        self._n_faults = self._schema.num_fault_classes()

        # Precompute static graph topology (doesn't change between timesteps)
        edge_pairs = self._schema.edge_index()
        self._edge_index = torch.tensor(
            [[s for s, t in edge_pairs], [t for s, t in edge_pairs]],
            dtype=torch.long,
        )
        weights = self._schema.edge_weights()
        rel_types = self._schema.edge_relation_types()
        rel_encoded = [self.RELATION_TYPE_MAP[r] for r in rel_types]
        self._edge_attr = torch.tensor(
            [[w, r] for w, r in zip(weights, rel_encoded)],
            dtype=torch.float,
        )
        self._node_id_to_idx = {
            n.id: i for i, n in enumerate(self._schema.NODES)
        }
        logger.info(
            f"GraphBuilder ready — {self._n_nodes} nodes, "
            f"{self._edge_index.shape[1]} edges"
        )

    def build(
        self,
        node_features: Dict[str, np.ndarray],
        window_ms: int,
        unit_id: str,
        ttf_hours: Optional[float] = None,
        fault_labels: Optional[List[int]] = None,
    ) -> Optional[Data]:
        """
        Build a PyG Data object for one timestep.

        Args:
            node_features: {node_id: embedding_array (node_feature_dim,)}
            window_ms: window centre timestamp in milliseconds
            unit_id: compressor unit identifier
            ttf_hours: ground-truth time-to-failure (training only)
            fault_labels: list of active fault class IDs (training only)

        Returns:
            PyG Data object, or None if any node feature is missing.
        """
        # Check all nodes are present
        missing = [n for n in self._schema.node_ids() if n not in node_features]
        if missing:
            logger.debug(f"Missing node features for window {window_ms}: {missing}")
            return None

        # Build node feature matrix
        x_rows = []
        for node in self._schema.NODES:
            feat = node_features[node.id]
            if feat.shape[0] != self._node_feature_dim:
                raise ValueError(
                    f"Node '{node.id}' feature dim mismatch: "
                    f"got {feat.shape[0]}, expected {self._node_feature_dim}"
                )
            x_rows.append(feat)

        x = torch.from_numpy(np.stack(x_rows, axis=0)).float()

        data = Data(
            x=x,
            edge_index=self._edge_index.clone(),
            edge_attr=self._edge_attr.clone(),
        )

        # Labels (training only)
        if ttf_hours is not None:
            data.y_ttf = torch.tensor([ttf_hours], dtype=torch.float)

        if fault_labels is not None:
            y_fault = torch.zeros(self._n_faults, dtype=torch.float)
            for fid in fault_labels:
                if 0 <= fid < self._n_faults:
                    y_fault[fid] = 1.0
            data.y_fault = y_fault

        data.window_ms = torch.tensor([window_ms], dtype=torch.long)
        data.unit_id = unit_id

        return data

    def build_temporal_sequence(
        self,
        sequence: List[Dict[str, np.ndarray]],
        window_ms_list: List[int],
        unit_id: str,
        ttf_hours: Optional[float] = None,
        fault_labels: Optional[List[int]] = None,
    ) -> Optional[List[Data]]:
        """
        Build a temporal sequence of graph snapshots.
        Returns None if any timestep is incomplete.

        This is the primary training input format:
        a list of T consecutive graph snapshots.
        """
        graphs = []
        for i, (feats, wms) in enumerate(zip(sequence, window_ms_list)):
            # Only attach labels to the final snapshot in the sequence
            g = self.build(
                node_features=feats,
                window_ms=wms,
                unit_id=unit_id,
                ttf_hours=ttf_hours if i == len(sequence) - 1 else None,
                fault_labels=fault_labels if i == len(sequence) - 1 else None,
            )
            if g is None:
                return None
            graphs.append(g)
        return graphs

    @property
    def num_nodes(self) -> int:
        return self._n_nodes

    @property
    def num_edges(self) -> int:
        return self._edge_index.shape[1]

    @property
    def edge_attr_dim(self) -> int:
        return self._edge_attr.shape[1]
