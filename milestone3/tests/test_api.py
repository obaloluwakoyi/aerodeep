"""
milestone3/tests/test_api.py
Integration tests for the FastAPI inference server.
"""

import numpy as np
import pytest
from httpx import AsyncClient
from fastapi.testclient import TestClient

from milestone3.api.main import app
from milestone2.graph.schema import CompressorGraphSchema


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def _dummy_request_payload(T: int = 10, ts_dim: int = 128, text_dim: int = 768) -> dict:
    node_ids = CompressorGraphSchema.node_ids()
    return {
        "unit_id": "C1001A",
        "window_ms": 1_710_000_000_000,
        "ts_sequence": {
            nid: np.random.randn(T, ts_dim).tolist()
            for nid in node_ids
        },
        "txt_embeddings": {
            nid: np.zeros(text_dim).tolist()
            for nid in node_ids
        },
        "fault_threshold": 0.45,
    }


class TestHealthEndpoint:
    def test_health_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "model_loaded" in data
        assert "device" in data


class TestGraphTopologyEndpoint:
    def test_topology_returns_nodes_and_edges(self, client):
        resp = client.get("/graph/topology")
        assert resp.status_code == 200
        data = resp.json()
        assert "nodes" in data
        assert "edges" in data
        assert len(data["nodes"]) == 6

    def test_all_node_fields_present(self, client):
        resp = client.get("/graph/topology")
        nodes = resp.json()["nodes"]
        for node in nodes:
            assert "id" in node
            assert "label" in node
            assert "sensors" in node


class TestFaultTaxonomyEndpoint:
    def test_taxonomy_returns_18_classes(self, client):
        resp = client.get("/faults/taxonomy")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["fault_classes"]) == 18

    def test_each_class_has_label(self, client):
        resp = client.get("/faults/taxonomy")
        for fc in resp.json()["fault_classes"]:
            assert "label" in fc
            assert len(fc["label"]) > 0


class TestDiagnoseEndpoint:
    def test_diagnose_returns_200(self, client):
        payload = _dummy_request_payload()
        resp = client.post("/diagnose", json=payload)
        assert resp.status_code == 200

    def test_diagnose_response_fields(self, client):
        payload = _dummy_request_payload()
        resp = client.post("/diagnose", json=payload)
        data = resp.json()
        assert "ttf_hours" in data
        assert "fault_probabilities" in data
        assert "node_risk" in data
        assert "top_faults" in data
        assert "inference_ms" in data

    def test_diagnose_ttf_non_negative(self, client):
        payload = _dummy_request_payload()
        resp = client.post("/diagnose", json=payload)
        assert resp.json()["ttf_hours"] >= 0.0

    def test_diagnose_node_risk_in_range(self, client):
        payload = _dummy_request_payload()
        resp = client.post("/diagnose", json=payload)
        for nid, risk in resp.json()["node_risk"].items():
            assert 0.0 <= risk <= 1.0, f"Node {nid} risk {risk} out of [0,1]"

    def test_diagnose_missing_node_returns_422(self, client):
        payload = _dummy_request_payload()
        del payload["ts_sequence"]["hp_cylinder"]
        resp = client.post("/diagnose", json=payload)
        assert resp.status_code == 422

    def test_diagnose_fault_probs_sum_not_constrained(self, client):
        # Multi-label: probabilities do NOT need to sum to 1
        payload = _dummy_request_payload()
        resp = client.post("/diagnose", json=payload)
        probs = resp.json()["fault_probabilities"]
        assert len(probs) == 18

    def test_diagnose_top_faults_sorted(self, client):
        payload = _dummy_request_payload()
        resp = client.post("/diagnose", json=payload)
        top = resp.json()["top_faults"]
        probs = [f["probability"] for f in top]
        assert probs == sorted(probs, reverse=True)
