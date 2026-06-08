"""
milestone3/dashboard/risk_panel.py

Risk migration panel: TTF gauge, node risk bar chart,
time-series risk history, and live alert feed.
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta
from typing import Dict, List

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from milestone2.graph.schema import CompressorGraphSchema


def render_ttf_gauge(ttf_hours: float, confidence: float) -> go.Figure:
    """Render a speedometer-style gauge for time-to-failure."""
    max_hours = 168  # 1 week

    color = "#22c55e"
    if ttf_hours < 24:
        color = "#ef4444"
    elif ttf_hours < 72:
        color = "#f59e0b"

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=ttf_hours,
        title={"text": "Time to Failure (hours)", "font": {"size": 16}},
        number={"suffix": " hrs", "font": {"size": 28}},
        gauge={
            "axis": {"range": [0, max_hours], "tickwidth": 1},
            "bar": {"color": color, "thickness": 0.25},
            "bgcolor": "white",
            "borderwidth": 2,
            "bordercolor": "gray",
            "steps": [
                {"range": [0, 24],  "color": "#fef2f2"},
                {"range": [24, 72], "color": "#fefce8"},
                {"range": [72, max_hours], "color": "#f0fdf4"},
            ],
            "threshold": {
                "line": {"color": "#dc2626", "width": 4},
                "thickness": 0.75,
                "value": 24,
            },
        },
    ))
    fig.update_layout(height=260, margin=dict(l=20, r=20, t=60, b=20))
    return fig


def render_node_risk_bar(node_risk: Dict[str, float]) -> go.Figure:
    """Horizontal bar chart of per-node risk."""
    schema = CompressorGraphSchema
    labels = {n.id: n.label for n in schema.NODES}

    df = pd.DataFrame([
        {"node": labels.get(nid, nid), "risk": risk, "node_id": nid}
        for nid, risk in sorted(node_risk.items(), key=lambda x: x[1], reverse=True)
    ])

    colors = []
    for _, row in df.iterrows():
        if row["risk"] < 0.30:
            colors.append("#22c55e")
        elif row["risk"] < 0.60:
            colors.append("#f59e0b")
        elif row["risk"] < 0.85:
            colors.append("#ef4444")
        else:
            colors.append("#7f1d1d")

    fig = go.Figure(go.Bar(
        x=df["risk"],
        y=df["node"],
        orientation="h",
        marker_color=colors,
        text=[f"{r:.0%}" for r in df["risk"]],
        textposition="outside",
    ))
    fig.update_layout(
        title="Node Risk Scores",
        xaxis=dict(range=[0, 1.1], title="Risk Score"),
        yaxis=dict(autorange="reversed"),
        height=300,
        margin=dict(l=10, r=40, t=40, b=20),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def render_risk_history_chart(history: List[Dict]) -> go.Figure:
    """Line chart of risk score over time for each node."""
    if not history:
        return go.Figure()

    schema = CompressorGraphSchema
    labels = {n.id: n.label for n in schema.NODES}

    fig = go.Figure()
    node_ids = CompressorGraphSchema.node_ids()

    color_map = {
        "lp_cylinder": "#3b82f6",
        "intercooler": "#8b5cf6",
        "hp_cylinder": "#ef4444",
        "shaft_coupling": "#f59e0b",
        "lube_oil_system": "#10b981",
        "seal_system": "#6366f1",
    }

    for nid in node_ids:
        times = [h.get("timestamp", i) for i, h in enumerate(history)]
        risks = [h.get("node_risk", {}).get(nid, 0.0) for h in history]
        fig.add_trace(go.Scatter(
            x=times,
            y=risks,
            name=labels.get(nid, nid),
            line=dict(color=color_map.get(nid, "#6b7280"), width=2),
            mode="lines",
        ))

    fig.add_hline(y=0.60, line_dash="dash", line_color="#f59e0b",
                  annotation_text="High risk threshold")
    fig.add_hline(y=0.85, line_dash="dash", line_color="#ef4444",
                  annotation_text="Critical threshold")

    fig.update_layout(
        title="Risk Migration Over Time",
        xaxis_title="Time",
        yaxis_title="Risk Score",
        yaxis=dict(range=[0, 1.05]),
        height=350,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def render_risk_panel():
    st.title("📊 Risk Migration Monitor")
    st.caption(f"Unit: **{st.session_state.selected_unit}**")

    diagnostic = st.session_state.get("last_diagnostic")

    # ── KPI row ───────────────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)

    ttf = diagnostic["ttf_hours"] if diagnostic else 168.0
    conf = diagnostic["ttf_confidence"] if diagnostic else 0.5

    with col1:
        st.metric(
            "Time to Failure",
            f"{ttf:.1f} hrs",
            delta=f"{'⚠️ < 24h ALERT' if ttf < 24 else '✅ Nominal'}",
            delta_color="inverse" if ttf < 24 else "normal",
        )
    with col2:
        node_risk = diagnostic["node_risk"] if diagnostic else {}
        highest_risk = max(node_risk.values()) if node_risk else 0.0
        st.metric("Highest Node Risk", f"{highest_risk:.0%}")
    with col3:
        active_faults = sum(
            1 for v in (diagnostic or {}).get("fault_probabilities", {}).values()
            if v > 0.45
        )
        st.metric("Active Fault Signals", active_faults)
    with col4:
        st.metric("Prediction Confidence", f"{conf:.0%}")

    st.divider()

    # ── Main charts ───────────────────────────────────────────────────────────
    col_gauge, col_bars = st.columns([1, 1])

    with col_gauge:
        st.plotly_chart(render_ttf_gauge(ttf, conf), use_container_width=True)

    with col_bars:
        if diagnostic and diagnostic.get("node_risk"):
            st.plotly_chart(
                render_node_risk_bar(diagnostic["node_risk"]),
                use_container_width=True,
            )
        else:
            st.info("No diagnostic data available. Run a diagnosis first.")

    # ── Risk history ──────────────────────────────────────────────────────────
    history = st.session_state.get("risk_history", [])
    if history:
        st.plotly_chart(render_risk_history_chart(history), use_container_width=True)

    # ── Alert feed ────────────────────────────────────────────────────────────
    st.subheader("🚨 Recent Alerts")
    alerts = st.session_state.get("alert_history", [])
    if alerts:
        for alert in reversed(alerts[-10:]):
            severity = alert.get("severity", "INFO")
            icon = "🚨" if severity == "CRITICAL" else "⚠️" if severity == "HIGH" else "ℹ️"
            st.markdown(
                f"{icon} `{alert.get('timestamp', '')}` — "
                f"**{alert.get('node', '')}**: {alert.get('message', '')}"
            )
    else:
        st.success("No active alerts")
