"""
milestone3/api/schemas.py
Pydantic request/response models for the AeroDeep inference API.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class DiagnoseRequest(BaseModel):
    unit_id: str = Field(..., description="Compressor unit identifier e.g. C1001A")
    window_ms: int = Field(..., description="Window centre timestamp in UNIX milliseconds")

    # {node_id: [[T timesteps × ts_dim features]]} — shape (T, ts_dim) per node
    ts_sequence: Dict[str, List[List[float]]] = Field(
        ..., description="Time-series embeddings per node: {node_id: (T, ts_dim)}"
    )

    # {node_id: [embed_dim floats]} — text embedding from vector store retrieval
    txt_embeddings: Dict[str, List[float]] = Field(
        ..., description="Text embeddings per node: {node_id: (text_dim,)}"
    )

    fault_threshold: Optional[float] = Field(
        default=0.45, ge=0.0, le=1.0,
        description="Sigmoid threshold for fault activation"
    )


class FaultPrediction(BaseModel):
    label: str
    probability: float


class DiagnoseResponse(BaseModel):
    unit_id: str
    window_ms: int

    # TTF
    ttf_hours: float = Field(..., description="Predicted hours until next failure event")
    ttf_confidence: float = Field(..., description="Confidence score [0,1]")

    # Faults
    fault_probabilities: Dict[str, float] = Field(
        ..., description="All fault class probabilities"
    )
    top_faults: List[FaultPrediction] = Field(
        ..., description="Top-5 most probable fault classes"
    )

    # Graph viz
    node_risk: Dict[str, float] = Field(
        ..., description="Per-node risk scores [0,1] for dashboard"
    )
    text_gate_values: Dict[str, float] = Field(
        ..., description="Text gate weight per node (0=ts-only, 1=full log context)"
    )

    inference_ms: float = Field(..., description="Server-side inference time in ms")


class GraphTopologyResponse(BaseModel):
    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]


class FaultTaxonomyResponse(BaseModel):
    fault_classes: List[Dict[str, Any]]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
