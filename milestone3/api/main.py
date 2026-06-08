"""
milestone3/api/main.py

FastAPI inference server for the AeroDeep diagnostic system.

Endpoints:
  POST /diagnose          — full diagnostic for a sensor + log batch
  GET  /health            — liveness probe
  GET  /graph/topology    — compressor graph structure for dashboard
  GET  /faults/taxonomy   — fault class definitions
"""

from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import yaml
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from milestone3.api.schemas import (
    DiagnoseRequest,
    DiagnoseResponse,
    GraphTopologyResponse,
    FaultTaxonomyResponse,
    HealthResponse,
)
from milestone3.model.dual_head import AeroDeepDiagnosticModel
from milestone2.graph.schema import CompressorGraphSchema
from milestone2.graph.builder import CompressorGraphBuilder


# ── Application state ─────────────────────────────────────────────────────────

class AppState:
    model: Optional[AeroDeepDiagnosticModel] = None
    builder: Optional[CompressorGraphBuilder] = None
    edge_index: Optional[torch.Tensor] = None
    edge_attr: Optional[torch.Tensor] = None
    device: str = "cpu"
    cfg: dict = {}


state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, clean up on shutdown."""
    cfg_path = os.environ.get("AERODEEP_CONFIG", "configs/config.yaml")
    ckpt_path = os.environ.get("AERODEEP_CHECKPOINT", "checkpoints/best_model.ckpt")

    with open(cfg_path) as f:
        state.cfg = yaml.safe_load(f)

    model_cfg = state.cfg["model"]
    state.device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Inference device: {state.device}")

    state.model = AeroDeepDiagnosticModel(
        timeseries_dim=model_cfg["timeseries_dim"],
        text_dim=model_cfg["text_dim"],
        fusion_dim=model_cfg["fusion_dim"],
        hidden_channels=model_cfg["hidden_channels"],
        n_fault_classes=CompressorGraphSchema.num_fault_classes(),
        dropout=0.0,  # disabled at inference
    ).to(state.device)

    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=state.device)
        # Handle Lightning checkpoint format
        state_dict = ckpt.get("state_dict", ckpt)
        # Strip "model." prefix added by Lightning
        clean_sd = {k.replace("model.", "", 1): v for k, v in state_dict.items()}
        state.model.load_state_dict(clean_sd, strict=False)
        logger.info(f"Loaded checkpoint: {ckpt_path}")
    else:
        logger.warning(f"Checkpoint not found at {ckpt_path} — using random weights")

    state.model.eval()

    state.builder = CompressorGraphBuilder(
        node_feature_dim=model_cfg["fusion_dim"]
    )

    # Precompute graph topology tensors
    dummy_feats = {
        nid: np.zeros(model_cfg["fusion_dim"], dtype=np.float32)
        for nid in CompressorGraphSchema.node_ids()
    }
    dummy_data = state.builder.build(dummy_feats, window_ms=0, unit_id="init")
    state.edge_index = dummy_data.edge_index.to(state.device)
    state.edge_attr = dummy_data.edge_attr.to(state.device)

    logger.info("AeroDeep inference server ready")
    yield

    logger.info("Shutting down inference server")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="AeroDeep Fault Diagnostic API",
    description="Multimodal fault diagnosis for offshore compressor assets",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok",
        model_loaded=state.model is not None,
        device=state.device,
    )


@app.get("/graph/topology", response_model=GraphTopologyResponse)
async def graph_topology():
    nodes = [
        {
            "id": n.id,
            "label": n.label,
            "description": n.description,
            "sensors": n.sensors,
            "node_type": n.node_type,
            "criticality": n.criticality,
            "viz_x": n.viz_x,
            "viz_y": n.viz_y,
        }
        for n in CompressorGraphSchema.NODES
    ]
    edges = [
        {
            "source": e.source,
            "target": e.target,
            "relation": e.relation.value,
            "weight": e.weight,
            "bidirectional": e.bidirectional,
        }
        for e in CompressorGraphSchema.EDGES
    ]
    return GraphTopologyResponse(nodes=nodes, edges=edges)


@app.get("/faults/taxonomy", response_model=FaultTaxonomyResponse)
async def fault_taxonomy():
    return FaultTaxonomyResponse(
        fault_classes=[
            {
                "id": f["id"],
                "node": f["node"],
                "fault": f["fault"],
                "label": CompressorGraphSchema.fault_label_by_id(f["id"]),
            }
            for f in CompressorGraphSchema.FAULT_CLASSES
        ]
    )


@app.post("/diagnose", response_model=DiagnoseResponse)
async def diagnose(request: DiagnoseRequest):
    if state.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    t0 = time.perf_counter()

    # Convert request payloads to numpy
    ts_sequence = {
        nid: np.array(arr, dtype=np.float32)
        for nid, arr in request.ts_sequence.items()
    }
    txt_embeddings = {
        nid: np.array(arr, dtype=np.float32)
        for nid, arr in request.txt_embeddings.items()
    }

    # Validate node coverage
    expected_nodes = set(CompressorGraphSchema.node_ids())
    missing_ts = expected_nodes - set(ts_sequence.keys())
    if missing_ts:
        raise HTTPException(
            status_code=422,
            detail=f"Missing time-series embeddings for nodes: {missing_ts}",
        )

    # Run inference
    try:
        result = state.model.predict(
            ts_sequence=ts_sequence,
            txt_embeddings=txt_embeddings,
            edge_index=state.edge_index,
            edge_attr=state.edge_attr,
            fault_threshold=request.fault_threshold or 0.45,
            unit_id=request.unit_id,
            window_ms=request.window_ms,
        )
    except Exception as exc:
        logger.error(f"Inference error for unit {request.unit_id}: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))

    elapsed_ms = (time.perf_counter() - t0) * 1000

    return DiagnoseResponse(
        unit_id=result.unit_id,
        window_ms=result.window_ms,
        ttf_hours=result.ttf_hours,
        ttf_confidence=result.ttf_confidence,
        fault_probabilities=result.fault_probabilities,
        top_faults=[
            {"label": label, "probability": prob}
            for label, prob in result.top_faults
        ],
        node_risk=result.node_risk,
        text_gate_values=result.text_gate_values,
        inference_ms=elapsed_ms,
    )
