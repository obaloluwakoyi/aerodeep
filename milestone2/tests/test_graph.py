"""
milestone2/tests/test_graph.py
Tests for graph construction and ST-GCN forward pass.
"""

import numpy as np
import pytest
import torch

from milestone2.graph.schema import CompressorGraphSchema
from milestone2.graph.builder import CompressorGraphBuilder
from milestone2.fusion.stgcn import AeroDeepSTGCN, STGCNEncoder


class TestCompressorGraphSchema:
    def test_node_count(self):
        assert len(CompressorGraphSchema.NODES) == 6

    def test_all_node_ids_unique(self):
        ids = CompressorGraphSchema.node_ids()
        assert len(ids) == len(set(ids))

    def test_sensor_to_node_map_covers_all_sensors(self):
        mapping = CompressorGraphSchema.sensor_to_node_map()
        all_sensors = [s for n in CompressorGraphSchema.NODES for s in n.sensors]
        for s in all_sensors:
            assert s in mapping

    def test_edge_index_shape(self):
        edges = CompressorGraphSchema.edge_index()
        assert isinstance(edges, list)
        assert all(len(e) == 2 for e in edges)

    def test_fault_classes_unique_ids(self):
        ids = [f["id"] for f in CompressorGraphSchema.FAULT_CLASSES]
        assert len(ids) == len(set(ids))

    def test_fault_label_by_id(self):
        label = CompressorGraphSchema.fault_label_by_id(0)
        assert "hp_cylinder" in label
        assert "valve" in label


class TestCompressorGraphBuilder:
    def setup_method(self):
        self.fusion_dim = 256
        self.builder = CompressorGraphBuilder(node_feature_dim=self.fusion_dim)

    def _dummy_node_features(self):
        return {
            nid: np.random.randn(self.fusion_dim).astype(np.float32)
            for nid in CompressorGraphSchema.node_ids()
        }

    def test_build_returns_data(self):
        feats = self._dummy_node_features()
        data = self.builder.build(feats, window_ms=1000, unit_id="C1001")
        assert data is not None

    def test_build_node_feature_shape(self):
        feats = self._dummy_node_features()
        data = self.builder.build(feats, window_ms=1000, unit_id="C1001")
        assert data.x.shape == (6, self.fusion_dim)

    def test_build_edge_index_shape(self):
        feats = self._dummy_node_features()
        data = self.builder.build(feats, window_ms=1000, unit_id="C1001")
        assert data.edge_index.shape[0] == 2

    def test_build_with_labels(self):
        feats = self._dummy_node_features()
        data = self.builder.build(
            feats, window_ms=1000, unit_id="C1001",
            ttf_hours=12.5, fault_labels=[0, 4]
        )
        assert data.y_ttf is not None
        assert data.y_fault is not None
        assert data.y_ttf.item() == pytest.approx(12.5)
        assert data.y_fault[0] == 1.0
        assert data.y_fault[4] == 1.0
        assert data.y_fault[1] == 0.0

    def test_build_missing_node_returns_none(self):
        feats = self._dummy_node_features()
        del feats["hp_cylinder"]
        data = self.builder.build(feats, window_ms=1000, unit_id="C1001")
        assert data is None

    def test_build_temporal_sequence(self):
        seq = [self._dummy_node_features() for _ in range(10)]
        wms = list(range(0, 10000, 1000))
        graphs = self.builder.build_temporal_sequence(
            seq, wms, unit_id="C1001", ttf_hours=5.0, fault_labels=[2]
        )
        assert graphs is not None
        assert len(graphs) == 10
        assert graphs[-1].y_ttf is not None
        assert graphs[0].y_ttf is None


class TestSTGCN:
    def setup_method(self):
        self.N = 6
        self.F = 256
        self.T = 15
        self.n_faults = 18
        self.model = AeroDeepSTGCN(
            node_feature_dim=self.F,
            n_fault_classes=self.n_faults,
            hidden_channels=[64, 32, 16],
            temporal_kernel_size=3,
            heads=2,
        )
        self.model.eval()

        self.builder = CompressorGraphBuilder(node_feature_dim=self.F)
        dummy_feats = {
            nid: np.random.randn(self.F).astype(np.float32)
            for nid in CompressorGraphSchema.node_ids()
        }
        data = self.builder.build(dummy_feats, window_ms=0, unit_id="C1001")
        self.edge_index = data.edge_index
        self.edge_attr = data.edge_attr

    def _make_sequence(self):
        return torch.randn(self.T, self.N, self.F)

    def test_output_shapes(self):
        x = self._make_sequence()
        with torch.no_grad():
            ttf, fault_logits, node_risk = self.model(x, self.edge_index, self.edge_attr)
        assert ttf.shape == (1,)
        assert fault_logits.shape == (self.n_faults,)
        assert node_risk.shape == (self.N,)

    def test_node_risk_in_range(self):
        x = self._make_sequence()
        with torch.no_grad():
            _, _, node_risk = self.model(x, self.edge_index, self.edge_attr)
        assert (node_risk >= 0).all()
        assert (node_risk <= 1).all()

    def test_gradients_flow(self):
        self.model.train()
        x = self._make_sequence().requires_grad_(True)
        ttf, fault_logits, _ = self.model(x, self.edge_index, self.edge_attr)
        loss = ttf.sum() + fault_logits.sum()
        loss.backward()
        assert x.grad is not None
