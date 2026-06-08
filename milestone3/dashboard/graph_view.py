"""
milestone3/dashboard/graph_view.py

Live interactive asset graph panel.
Renders the compressor graph as an interactive network diagram
where nodes transition green → amber → red as risk scores rise.
Uses streamlit-agraph for interactive graph rendering.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional

import httpx
import streamlit as st
from streamlit_agraph import agraph, Node, Edge, Config


# ── Colour mapping ────────────────────────────────────────────────────────────

def risk_to_color(risk: float) -> str:
    """Map risk score [0,1] to hex colour."""
    if risk < 0.30:
        return "#22c55e"    # green
    elif risk < 0.60:
        return "#f59e0b"    # amber
    elif risk < 0.85:
        return "#ef4444"    # red
    else:
        return "#7f1d1d"    # dark red — critical


def risk_to_size(risk: float, base: int = 22) -> int:
    """Larger nodes = higher risk (visual urgency cue)."""
    return int(base + risk * 18)


def risk_to_label(risk: float) -> str:
    if risk < 0.30:
        return "✅ Normal"
    elif risk < 0.60:
        return "⚠️ Elevated"
    elif risk < 0.85:
        return "🔴 High Risk"
    else:
        return "🚨 Critical"


# ── Graph builder ─────────────────────────────────────────────────────────────

def build_agraph(
    topology: dict,
    node_risk: Dict[str, float],
) -> tuple:
    nodes = []
    for n in topology["nodes"]:
        nid = n["id"]
        risk = node_risk.get(nid, 0.0)
        color = risk_to_color(risk)
        size = risk_to_size(risk)
        label = f"{n['label']}\n{risk_to_label(risk)}\n{risk:.0%}"

        nodes.append(Node(
            id=nid,
            label=label,
            size=size,
            color=color,
            title=f"{n['description']}\nSensors: {', '.join(n['sensors'])}",
            font={"size": 11, "color": "#ffffff"},
            shape="ellipse",
        ))

    edges = []
    relation_colors = {
        "fluid_flow": "#3b82f6",
        "mechanical_drive": "#8b5cf6",
        "lubrication": "#10b981",
        "sealing": "#f59e0b",
        "thermal_proximity": "#ef4444",
        "structural_proximity": "#6b7280",
    }
    for e in topology["edges"]:
        edges.append(Edge(
            source=e["source"],
            target=e["target"],
            label=e["relation"].replace("_", " "),
            color=relation_colors.get(e["relation"], "#6b7280"),
            width=int(e["weight"] * 3 + 1),
            arrows="to",
            dashes=e["relation"] in ("thermal_proximity", "structural_proximity"),
        ))

    config = Config(
        width="100%",
        height=520,
        directed=True,
        physics=True,
        hierarchical=False,
        nodeHighlightBehavior=True,
        highlightColor="#f1f5f9",
        collapsible=False,
        node={"labelProperty": "label"},
        link={"labelProperty": "label", "renderLabel": True},
    )

    return nodes, edges, config


# ── Main render ───────────────────────────────────────────────────────────────

def render_graph_view():
    st.title("⚙️ Live Asset Graph — Compressor Unit")
    st.caption(f"Unit: **{st.session_state.selected_unit}** | Auto-refresh every 5 seconds")

    api_url = st.session_state.api_url
    unit_id = st.session_state.selected_unit

    col_graph, col_legend = st.columns([3, 1])

    with col_legend:
        st.markdown("### Risk Legend")
        st.markdown("🟢 **Normal** < 30%")
        st.markdown("🟡 **Elevated** 30–60%")
        st.markdown("🔴 **High Risk** 60–85%")
        st.markdown("🚨 **Critical** > 85%")
        st.divider()
        st.markdown("### Edge Types")
        st.markdown("🔵 Fluid flow")
        st.markdown("🟣 Mechanical drive")
        st.markdown("🟢 Lubrication")
        st.markdown("🟡 Sealing")
        st.markdown("🔴 Thermal proximity")
        st.markdown("⚫ Structural")

    with col_graph:
        # Load graph topology
        try:
            resp = httpx.get(f"{api_url}/graph/topology", timeout=5.0)
            topology = resp.json()
        except Exception as e:
            st.error(f"Cannot reach API: {e}")
            return

        # Load current risk from last diagnostic (or zeros)
        diagnostic = st.session_state.get("last_diagnostic")
        if diagnostic and diagnostic.get("unit_id") == unit_id:
            node_risk = diagnostic.get("node_risk", {})
        else:
            node_risk = {n["id"]: 0.05 for n in topology["nodes"]}

        nodes, edges, config = build_agraph(topology, node_risk)

        # Render interactive graph
        selected = agraph(nodes=nodes, edges=edges, config=config)

        if selected:
            st.info(f"Selected node: **{selected}**")
            node_def = next(
                (n for n in topology["nodes"] if n["id"] == selected), None
            )
            if node_def:
                risk = node_risk.get(selected, 0.0)
                st.markdown(f"**{node_def['label']}**")
                st.markdown(f"_{node_def['description']}_")
                st.markdown(f"Sensors: `{', '.join(node_def['sensors'])}`")
                st.progress(risk, text=f"Risk: {risk:.1%}")

    # Auto-refresh
    refresh = st.empty()
    with refresh.container():
        if st.button("🔄 Refresh Now"):
            st.rerun()
    time.sleep(5)
    st.rerun()
