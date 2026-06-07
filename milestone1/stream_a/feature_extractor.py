"""
milestone1/stream_a/feature_extractor.py

Aggregates per-sensor feature vectors into per-node time-series embeddings.
Each compressor node (LP Cylinder, Intercooler, etc.) may have multiple
associated sensors. This module pools them into a single dense vector
ready for node-level fusion with the text embedding.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from loguru import logger

from milestone1.stream_a.preprocessor import SensorFeatureVector, TOTAL_FEATURES


@dataclass
class NodeTimeSeriesEmbedding:
    """
    Dense time-series embedding for one graph node at one time window.
    """
    unit_id: str
    node_id: str
    window_start_ms: int
    window_end_ms: int
    embedding: np.ndarray       # shape (timeseries_dim,)


class SensorPoolingEncoder(nn.Module):
    """
    Small MLP encoder that:
      1. Takes the raw feature vector for each sensor associated with a node
      2. Linearly projects all sensor vectors to a shared dim
      3. Pools across sensors (mean + max concatenation)
      4. Projects to final embedding dimension

    Input:  (n_sensors, TOTAL_FEATURES)
    Output: (timeseries_dim,)
    """

    def __init__(
        self,
        input_dim: int = TOTAL_FEATURES,
        hidden_dim: int = 128,
        output_dim: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.sensor_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # After mean+max pooling the hidden dim doubles
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (n_sensors, input_dim)
        Returns:
            (output_dim,)
        """
        projected = self.sensor_proj(x)          # (n_sensors, hidden_dim)
        mean_pool = projected.mean(dim=0)         # (hidden_dim,)
        max_pool = projected.max(dim=0).values    # (hidden_dim,)
        pooled = torch.cat([mean_pool, max_pool]) # (hidden_dim * 2,)
        return self.fusion(pooled)                # (output_dim,)


class NodeFeatureExtractor:
    """
    Manages a SensorPoolingEncoder per graph node.
    Accepts buffered SensorFeatureVectors and emits NodeTimeSeriesEmbeddings.

    In production this runs as a stateful processor downstream of the
    MultiSensorPreprocessor. It buffers per-sensor feature vectors until
    a complete set is available for a given (node, window_start_ms),
    then encodes and emits.
    """

    def __init__(
        self,
        node_sensor_map: Dict[str, List[str]],
        output_dim: int = 128,
        device: str = "cpu",
    ):
        """
        Args:
            node_sensor_map: maps node_id -> list of sensor_ids
              e.g. {"lp_cylinder": ["LP_PRESSURE", "LP_TEMPERATURE", ...]}
            output_dim: dimension of the output embedding
            device: torch device string
        """
        self._node_sensor_map = node_sensor_map
        self._output_dim = output_dim
        self._device = torch.device(device)

        # One encoder per node
        self._encoders: Dict[str, SensorPoolingEncoder] = {
            node_id: SensorPoolingEncoder(
                input_dim=TOTAL_FEATURES,
                output_dim=output_dim,
            ).to(self._device).eval()
            for node_id in node_sensor_map
        }

        # Accumulation buffers: node_id -> window_start_ms -> {sensor_id: vector}
        self._buffers: Dict[str, Dict[int, Dict[str, np.ndarray]]] = {
            node_id: {} for node_id in node_sensor_map
        }

        # Build reverse map: sensor_id -> node_id
        self._sensor_to_node: Dict[str, str] = {}
        for node_id, sensors in node_sensor_map.items():
            for sid in sensors:
                self._sensor_to_node[sid] = node_id

        logger.info(
            f"NodeFeatureExtractor ready — "
            f"{len(node_sensor_map)} nodes, output_dim={output_dim}"
        )

    def load_weights(self, checkpoint_path: str) -> None:
        """Load pretrained encoder weights from a checkpoint."""
        ckpt = torch.load(checkpoint_path, map_location=self._device)
        for node_id, state in ckpt.items():
            if node_id in self._encoders:
                self._encoders[node_id].load_state_dict(state)
        logger.info(f"Loaded encoder weights from {checkpoint_path}")

    def push(
        self, fv: SensorFeatureVector
    ) -> Optional[NodeTimeSeriesEmbedding]:
        """
        Accept a sensor feature vector. Returns a NodeTimeSeriesEmbedding
        when all sensors for that node's window have been received.
        """
        node_id = self._sensor_to_node.get(fv.sensor_id)
        if node_id is None:
            return None

        wkey = fv.window_start_ms
        if wkey not in self._buffers[node_id]:
            self._buffers[node_id][wkey] = {}

        self._buffers[node_id][wkey][fv.sensor_id] = fv.features

        required = set(self._node_sensor_map[node_id])
        received = set(self._buffers[node_id][wkey].keys())

        if not required.issubset(received):
            return None  # not all sensors complete yet

        # All sensors present — encode
        sensor_vectors = np.stack(
            [self._buffers[node_id][wkey][sid] for sid in self._node_sensor_map[node_id]],
            axis=0,
        )   # (n_sensors, TOTAL_FEATURES)

        # Clean up buffer
        del self._buffers[node_id][wkey]

        embedding = self._encode(node_id, sensor_vectors, fv.window_end_ms)
        return NodeTimeSeriesEmbedding(
            unit_id=fv.unit_id,
            node_id=node_id,
            window_start_ms=fv.window_start_ms,
            window_end_ms=fv.window_end_ms,
            embedding=embedding,
        )

    def _encode(
        self, node_id: str, sensor_matrix: np.ndarray, window_end_ms: int
    ) -> np.ndarray:
        encoder = self._encoders[node_id]
        x = torch.from_numpy(sensor_matrix).float().to(self._device)
        with torch.no_grad():
            emb = encoder(x)
        return emb.cpu().numpy()

    def get_node_embedding_dim(self) -> int:
        return self._output_dim
