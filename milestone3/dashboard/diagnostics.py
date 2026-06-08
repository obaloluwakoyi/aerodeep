"""
milestone3/dashboard/diagnostics.py

Root Cause Diagnostics panel.
Shows fault probability rankings, SHAP-style node attribution,
matched historical log entries, and plain-language explanations.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import plotly.graph_objects as go
import streamlit as st

from milestone2.graph.schema import CompressorGraphSchema


def render_fault_probability_chart(
    fault_probs: Dict[str, float],
    threshold: float = 0.45,
) -> go.Figure:
    """Horizontal bar chart of all fault class probabilities."""
    items = sorted(fault_probs.items(), key=lambda x: x[1], reverse=True)[:12]
    labels = [item[0] for item in items]
    values = [item[1] for item in items]

    colors = [
        "#ef4444" if v >= threshold else "#94a3b8"
        for v in values
    ]

    fig = go.Figure(go.Bar(
        x=values,
        y=labels,
        orientation="h",
        marker_color=colors,
        text=[f"{v:.0%}" for v in values],
        textposition="outside",
    ))
    fig.add_vline(
        x=threshold, line_dash="dash", line_color="#f59e0b",
        annotation_text=f"Threshold ({threshold:.0%})",
        annotation_position="top",
    )
    fig.update_layout(
        title="Fault Class Probabilities",
        xaxis=dict(range=[0, 1.15], title="Probability"),
        yaxis=dict(autorange="reversed"),
        height=420,
        margin=dict(l=10, r=50, t=50, b=20),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def render_node_attribution_chart(
    node_attribution: Dict[str, float],
) -> go.Figure:
    """Radar chart showing which nodes contributed to the diagnosis."""
    schema = CompressorGraphSchema
    labels_map = {n.id: n.label for n in schema.NODES}

    nids = list(node_attribution.keys())
    values = [node_attribution[nid] for nid in nids]
    labels = [labels_map.get(nid, nid) for nid in nids]
    values_closed = values + [values[0]]
    labels_closed = labels + [labels[0]]

    fig = go.Figure(go.Scatterpolar(
        r=values_closed,
        theta=labels_closed,
        fill="toself",
        fillcolor="rgba(239,68,68,0.2)",
        line=dict(color="#ef4444", width=2),
        name="Attribution",
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=False,
        title="Node Attribution (fault contribution)",
        height=350,
    )
    return fig


def _plain_language_explanation(
    top_faults: List[Tuple[str, float]],
    node_risk: Dict[str, float],
    ttf_hours: float,
) -> str:
    if not top_faults:
        return "No significant fault signals detected. System operating within normal parameters."

    top_label, top_prob = top_faults[0]
    highest_risk_node = max(node_risk.items(), key=lambda x: x[1]) if node_risk else ("unknown", 0.0)

    schema = CompressorGraphSchema
    labels_map = {n.id: n.label for n in schema.NODES}
    node_name = labels_map.get(highest_risk_node[0], highest_risk_node[0])

    urgency = "⚠️ Immediate inspection recommended" if ttf_hours < 24 else \
              "🔶 Plan maintenance within 3 days" if ttf_hours < 72 else \
              "📋 Schedule maintenance within 1 week"

    explanation = (
        f"**Primary diagnosis:** {top_label}\n\n"
        f"The model detects a **{top_prob:.0%} probability** of this fault condition. "
        f"The highest-risk component is the **{node_name}** (risk score: {highest_risk_node[1]:.0%}).\n\n"
        f"**Predicted time to failure:** {ttf_hours:.1f} hours\n\n"
        f"**Action:** {urgency}\n\n"
    )

    if len(top_faults) > 1:
        secondary = ", ".join(
            f"{label} ({prob:.0%})" for label, prob in top_faults[1:3]
        )
        explanation += f"**Secondary signals:** {secondary}"

    return explanation


def render_diagnostics():
    st.title("🔬 Root Cause Diagnostics")
    st.caption(f"Unit: **{st.session_state.selected_unit}**")

    diagnostic = st.session_state.get("last_diagnostic")

    if not diagnostic:
        st.info(
            "No diagnostic data loaded. Connect the live pipeline or upload "
            "a batch file to run a diagnosis."
        )
        # Demo mode button
        if st.button("▶️ Run Demo Diagnosis"):
            st.session_state.last_diagnostic = _demo_diagnostic()
            st.rerun()
        return

    # ── Plain-language summary ─────────────────────────────────────────────────
    st.subheader("📋 Diagnostic Summary")
    explanation = _plain_language_explanation(
        diagnostic.get("top_faults", []),
        diagnostic.get("node_risk", {}),
        diagnostic.get("ttf_hours", 168.0),
    )
    st.markdown(explanation)

    st.divider()

    # ── Fault probabilities + attribution ─────────────────────────────────────
    col_faults, col_attr = st.columns([3, 2])

    with col_faults:
        fault_probs = diagnostic.get("fault_probabilities", {})
        if fault_probs:
            st.plotly_chart(
                render_fault_probability_chart(fault_probs),
                use_container_width=True,
            )

    with col_attr:
        # Use node_risk as proxy for attribution (real version uses NodeAttribution)
        node_risk = diagnostic.get("node_risk", {})
        if node_risk:
            st.plotly_chart(
                render_node_attribution_chart(node_risk),
                use_container_width=True,
            )

    st.divider()

    # ── Text gate values ───────────────────────────────────────────────────────
    gates = diagnostic.get("text_gate_values", {})
    if gates:
        st.subheader("📄 Maintenance Log Contribution")
        st.caption(
            "How much historical maintenance log context influenced the diagnosis per node. "
            "Higher = log data significantly contributed."
        )
        schema = CompressorGraphSchema
        labels_map = {n.id: n.label for n in schema.NODES}

        cols = st.columns(len(gates))
        for i, (nid, gate_val) in enumerate(gates.items()):
            with cols[i]:
                st.metric(
                    labels_map.get(nid, nid),
                    f"{gate_val:.0%}",
                    help=f"Text gate for {nid}",
                )

    st.divider()

    # ── Top matched logs ───────────────────────────────────────────────────────
    st.subheader("📂 Similar Historical Cases")
    st.caption("Retrieved from maintenance log vector store (semantic similarity)")

    # Placeholder — in production these come from VectorStore.search_chunks()
    similar_logs = diagnostic.get("similar_logs", [])
    if similar_logs:
        for log in similar_logs[:3]:
            with st.expander(f"📄 {log.get('source', 'Log')} — similarity: {log.get('similarity', 0):.0%}"):
                st.markdown(log.get("text", ""))
    else:
        st.info("No historical matches loaded. Connect to pgvector store for semantic retrieval.")

    # ── Export ─────────────────────────────────────────────────────────────────
    st.divider()
    if st.button("📥 Export Diagnostic Report"):
        import json
        report = json.dumps(diagnostic, indent=2, default=str)
        st.download_button(
            label="Download JSON",
            data=report,
            file_name=f"aerodeep_diagnostic_{diagnostic.get('unit_id', 'unknown')}_{diagnostic.get('window_ms', 0)}.json",
            mime="application/json",
        )


def _demo_diagnostic() -> dict:
    """Returns a realistic demo diagnostic payload for the UI."""
    return {
        "unit_id": "C1001A",
        "window_ms": 1_710_000_000_000,
        "ttf_hours": 18.5,
        "ttf_confidence": 0.73,
        "fault_probabilities": {
            "hp_cylinder — valve leakage carbon deposit accumulation": 0.82,
            "hp_cylinder — piston ring wear": 0.41,
            "lube oil system — filter blockage": 0.38,
            "lp_cylinder — valve leakage carbon deposit accumulation": 0.22,
            "intercooler — tube fouling scaling": 0.15,
            "seal system — seal face wear high leakage": 0.11,
            "shaft coupling — misalignment": 0.06,
            "hp_cylinder — rod drop crosshead wear": 0.05,
        },
        "top_faults": [
            ["hp_cylinder — valve leakage carbon deposit accumulation", 0.82],
            ["hp_cylinder — piston ring wear", 0.41],
            ["lube oil system — filter blockage", 0.38],
        ],
        "node_risk": {
            "lp_cylinder": 0.28,
            "intercooler": 0.19,
            "hp_cylinder": 0.87,
            "shaft_coupling": 0.34,
            "lube_oil_system": 0.51,
            "seal_system": 0.23,
        },
        "text_gate_values": {
            "lp_cylinder": 0.31,
            "intercooler": 0.12,
            "hp_cylinder": 0.74,
            "shaft_coupling": 0.22,
            "lube_oil_system": 0.58,
            "seal_system": 0.19,
        },
        "similar_logs": [
            {
                "source": "Maintenance Report — C1001A — 2023-11-14.pdf",
                "similarity": 0.91,
                "text": (
                    "HP cylinder valve inspection revealed heavy carbon deposit accumulation "
                    "on suction valve plates. Discharge temperature running 12°C above baseline. "
                    "Valves cleaned and lapped. Full valve change recommended at next T/D."
                ),
            },
            {
                "source": "Shift Log — 2024-02-03 Night Shift",
                "similarity": 0.78,
                "text": (
                    "HP stage discharge temperature alarm at 04:30. Operator noted "
                    "HP pressure differential reduced by ~8%. Suspect valve efficiency loss. "
                    "Performance test scheduled for day shift."
                ),
            },
        ],
        "inference_ms": 12.4,
    }
