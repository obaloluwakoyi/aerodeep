"""
milestone2/graph/schema.py

Defines the physical topology of the offshore compressor unit as a
directed graph. Nodes represent physical assets; edges encode
thermodynamic, mechanical, and structural relationships.

This schema is the single source of truth for graph construction.
It is loaded by the GNN and the dashboard alike.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple


class RelationType(str, Enum):
    """Physical relationship types between compressor components."""
    FLUID_FLOW = "fluid_flow"           # Process gas flows between nodes
    MECHANICAL_DRIVE = "mechanical_drive"  # Shaft transmits torque
    LUBRICATION = "lubrication"         # Oil circuit feeds component
    SEALING = "sealing"                 # Seal system protects component
    THERMAL_PROXIMITY = "thermal_proximity"  # Heat transfer via proximity
    STRUCTURAL_PROXIMITY = "structural_proximity"  # Vibration transmission


@dataclass
class NodeDefinition:
    """A physical asset in the compressor unit."""
    id: str
    label: str
    description: str
    sensors: List[str]          # sensor IDs feeding this node
    node_type: str              # "rotating" | "static" | "auxiliary"
    criticality: float          # 0-1: how critical to plant operation
    # Graph layout hints for visualisation
    viz_x: float = 0.0
    viz_y: float = 0.0


@dataclass
class EdgeDefinition:
    """A directed physical relationship between two nodes."""
    source: str
    target: str
    relation: RelationType
    weight: float               # 0-1: strength of coupling
    bidirectional: bool = False
    description: str = ""


class CompressorGraphSchema:
    """
    Defines the full node/edge schema for a two-stage reciprocating
    compressor unit (representative of offshore Gulf of Guinea assets).

    Nodes (6):
      - lp_cylinder: Low-Pressure Cylinder & piston-rod assembly
      - intercooler: Inter-stage cooler + KO drum
      - hp_cylinder: High-Pressure Cylinder & piston-rod assembly
      - shaft_coupling: Drive shaft + flexible coupling
      - lube_oil_system: Lube oil pump, reservoir, filters
      - seal_system: Dry gas seal / mechanical seal system

    Edges (10 directed relationships):
      Process gas path: lp_cylinder → intercooler → hp_cylinder
      Mechanical: shaft_coupling → lp_cylinder, shaft_coupling → hp_cylinder
      Lube oil: lube_oil_system → lp_cylinder, → hp_cylinder, → shaft_coupling
      Sealing: seal_system → hp_cylinder
      Thermal cross-links: hp_cylinder ↔ lp_cylinder (thermal proximity)
      Structural: lp_cylinder → shaft_coupling (vibration propagation)
    """

    NODES: List[NodeDefinition] = [
        NodeDefinition(
            id="lp_cylinder",
            label="Low-Pressure Cylinder",
            description=(
                "First-stage compression cylinder. Houses LP piston, piston rod, "
                "suction and discharge valves. Inlet ~1 bara, outlet ~4.5 bara."
            ),
            sensors=[
                "LP_PRESSURE", "LP_TEMPERATURE",
                "LP_BEARING_VIB_X", "LP_BEARING_VIB_Y",
            ],
            node_type="rotating",
            criticality=0.9,
            viz_x=0.15, viz_y=0.5,
        ),
        NodeDefinition(
            id="intercooler",
            label="Intercooler",
            description=(
                "Shell-and-tube heat exchanger between compression stages. "
                "Cools process gas to ~40°C before second-stage inlet. "
                "Includes KO drum for condensate separation."
            ),
            sensors=["INTERCOOLER_TEMP"],
            node_type="static",
            criticality=0.6,
            viz_x=0.5, viz_y=0.3,
        ),
        NodeDefinition(
            id="hp_cylinder",
            label="High-Pressure Cylinder",
            description=(
                "Second-stage compression cylinder. Higher thermal load, "
                "tighter valve clearances. Outlet ~18 bara. "
                "Most vulnerable node for valve and ring failures."
            ),
            sensors=[
                "HP_PRESSURE", "HP_TEMPERATURE",
                "HP_BEARING_VIB_X", "HP_BEARING_VIB_Y",
            ],
            node_type="rotating",
            criticality=0.95,
            viz_x=0.85, viz_y=0.5,
        ),
        NodeDefinition(
            id="shaft_coupling",
            label="Shaft Coupling",
            description=(
                "Flexible disc coupling connecting driver (motor/turbine) to "
                "compressor crankshaft. Speed sensor mounted on shaft."
            ),
            sensors=["SHAFT_RPM"],
            node_type="rotating",
            criticality=0.85,
            viz_x=0.5, viz_y=0.8,
        ),
        NodeDefinition(
            id="lube_oil_system",
            label="Lube Oil System",
            description=(
                "Forced-feed lube oil circuit. Main pump, auxiliary pump, "
                "reservoir, cooler, filters, and header. Feeds all rotating "
                "bearing journals and crossheads."
            ),
            sensors=["LUBE_OIL_PRESSURE"],
            node_type="auxiliary",
            criticality=0.95,
            viz_x=0.15, viz_y=0.8,
        ),
        NodeDefinition(
            id="seal_system",
            label="Mechanical Seal System",
            description=(
                "Dry gas seal (DGS) on HP cylinder outboard end. "
                "Prevents process gas escape. Seal gas differential pressure "
                "monitored continuously."
            ),
            sensors=["SEAL_GAS_FLOW"],
            node_type="auxiliary",
            criticality=0.9,
            viz_x=0.85, viz_y=0.8,
        ),
    ]

    EDGES: List[EdgeDefinition] = [
        # ── Process gas path ─────────────────────────────────────────────────
        EdgeDefinition(
            source="lp_cylinder", target="intercooler",
            relation=RelationType.FLUID_FLOW, weight=1.0,
            description="LP discharge gas flows to intercooler inlet",
        ),
        EdgeDefinition(
            source="intercooler", target="hp_cylinder",
            relation=RelationType.FLUID_FLOW, weight=1.0,
            description="Cooled gas flows from intercooler to HP suction",
        ),
        # ── Mechanical drive ──────────────────────────────────────────────────
        EdgeDefinition(
            source="shaft_coupling", target="lp_cylinder",
            relation=RelationType.MECHANICAL_DRIVE, weight=0.9,
            description="Crankshaft drives LP cylinder crosshead",
        ),
        EdgeDefinition(
            source="shaft_coupling", target="hp_cylinder",
            relation=RelationType.MECHANICAL_DRIVE, weight=0.9,
            description="Crankshaft drives HP cylinder crosshead",
        ),
        # ── Lubrication ───────────────────────────────────────────────────────
        EdgeDefinition(
            source="lube_oil_system", target="lp_cylinder",
            relation=RelationType.LUBRICATION, weight=0.8,
            description="Oil feeds LP crosshead, main bearing journals",
        ),
        EdgeDefinition(
            source="lube_oil_system", target="hp_cylinder",
            relation=RelationType.LUBRICATION, weight=0.8,
            description="Oil feeds HP crosshead, main bearing journals",
        ),
        EdgeDefinition(
            source="lube_oil_system", target="shaft_coupling",
            relation=RelationType.LUBRICATION, weight=0.7,
            description="Oil lubricates coupling spider and crankshaft bearings",
        ),
        # ── Sealing ───────────────────────────────────────────────────────────
        EdgeDefinition(
            source="seal_system", target="hp_cylinder",
            relation=RelationType.SEALING, weight=0.85,
            description="DGS seals HP cylinder process-end",
        ),
        # ── Thermal / structural cross-links ─────────────────────────────────
        EdgeDefinition(
            source="hp_cylinder", target="lp_cylinder",
            relation=RelationType.THERMAL_PROXIMITY, weight=0.4,
            bidirectional=True,
            description="Frame thermal coupling; HP heat migrates to LP end",
        ),
        EdgeDefinition(
            source="lp_cylinder", target="shaft_coupling",
            relation=RelationType.STRUCTURAL_PROXIMITY, weight=0.5,
            description="Frame vibration propagates LP → coupling",
        ),
    ]

    # ── Convenience lookups ────────────────────────────────────────────────────

    @classmethod
    def node_ids(cls) -> List[str]:
        return [n.id for n in cls.NODES]

    @classmethod
    def node_by_id(cls, node_id: str) -> Optional[NodeDefinition]:
        return next((n for n in cls.NODES if n.id == node_id), None)

    @classmethod
    def sensor_to_node_map(cls) -> Dict[str, str]:
        """Returns {sensor_id: node_id}."""
        mapping = {}
        for node in cls.NODES:
            for sensor in node.sensors:
                mapping[sensor] = node.id
        return mapping

    @classmethod
    def node_sensor_map(cls) -> Dict[str, List[str]]:
        """Returns {node_id: [sensor_ids]}."""
        return {node.id: node.sensors for node in cls.NODES}

    @classmethod
    def edge_index(cls) -> List[Tuple[int, int]]:
        """Returns edge list as (src_idx, tgt_idx) pairs for PyG."""
        id_to_idx = {n.id: i for i, n in enumerate(cls.NODES)}
        edges = []
        for e in cls.EDGES:
            src, tgt = id_to_idx[e.source], id_to_idx[e.target]
            edges.append((src, tgt))
            if e.bidirectional:
                edges.append((tgt, src))
        return edges

    @classmethod
    def edge_weights(cls) -> List[float]:
        """Edge attribute weights matching edge_index order."""
        weights = []
        for e in cls.EDGES:
            weights.append(e.weight)
            if e.bidirectional:
                weights.append(e.weight)
        return weights

    @classmethod
    def edge_relation_types(cls) -> List[str]:
        """Edge relation type strings matching edge_index order."""
        rel_types = []
        for e in cls.EDGES:
            rel_types.append(e.relation.value)
            if e.bidirectional:
                rel_types.append(e.relation.value)
        return rel_types

    # ── Fault taxonomy ────────────────────────────────────────────────────────

    FAULT_CLASSES: List[Dict] = [
        {"id": 0,  "node": "hp_cylinder",    "fault": "valve_leakage_carbon_deposit"},
        {"id": 1,  "node": "hp_cylinder",    "fault": "valve_failure_broken_plate"},
        {"id": 2,  "node": "hp_cylinder",    "fault": "piston_ring_wear"},
        {"id": 3,  "node": "hp_cylinder",    "fault": "packing_ring_leakage"},
        {"id": 4,  "node": "lp_cylinder",    "fault": "valve_leakage_carbon_deposit"},
        {"id": 5,  "node": "lp_cylinder",    "fault": "valve_failure_broken_plate"},
        {"id": 6,  "node": "lp_cylinder",    "fault": "piston_ring_wear"},
        {"id": 7,  "node": "lp_cylinder",    "fault": "packing_ring_leakage"},
        {"id": 8,  "node": "intercooler",    "fault": "tube_fouling_scaling"},
        {"id": 9,  "node": "intercooler",    "fault": "tube_leak_process_cooling_water"},
        {"id": 10, "node": "shaft_coupling", "fault": "misalignment"},
        {"id": 11, "node": "shaft_coupling", "fault": "coupling_spider_wear"},
        {"id": 12, "node": "lube_oil_system","fault": "pump_degradation_low_flow"},
        {"id": 13, "node": "lube_oil_system","fault": "filter_blockage"},
        {"id": 14, "node": "lube_oil_system","fault": "oil_contamination_water_ingress"},
        {"id": 15, "node": "seal_system",    "fault": "seal_face_wear_high_leakage"},
        {"id": 16, "node": "seal_system",    "fault": "seal_gas_contamination"},
        {"id": 17, "node": "hp_cylinder",    "fault": "rod_drop_crosshead_wear"},
    ]

    @classmethod
    def num_fault_classes(cls) -> int:
        return len(cls.FAULT_CLASSES)

    @classmethod
    def fault_label_by_id(cls, fault_id: int) -> str:
        for f in cls.FAULT_CLASSES:
            if f["id"] == fault_id:
                return f"{f['node']} — {f['fault'].replace('_', ' ')}"
        return "unknown"
