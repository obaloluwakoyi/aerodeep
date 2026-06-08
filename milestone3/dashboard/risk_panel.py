"""
risk_panel.py
Dashboard Visualization Panel mapping Key Performance Indicators and Risk Profiles.
"""
import streamlit as st
import pandas as pd

def render_risk_panel(diagnostic: dict):
    """
    Renders the risk panel dashboard UI using defensive parameters.
    Guarantees no KeyError triggers by employing strict default fallbacks.
    """
    st.subheader("📊 Asset Health Risk Monitoring Panel")
    
    # Extract structural state safely via fallback defaults
    ttf = diagnostic.get("ttf_hours", 168.0) if isinstance(diagnostic, dict) else 168.0
    confidence = diagnostic.get("ttf_confidence", 0.50) if isinstance(diagnostic, dict) else 0.50
    node_risks = diagnostic.get("node_risk", {}) if isinstance(diagnostic, dict) else {}
    
    # ── STRUCTURAL METRIC CARDS ──
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Estimated Time-to-Failure (TTF)",
            value=f"{ttf:.1f} Hours",
            delta="- Immediate Threat Window" if ttf < 24.0 else "Stable Operating Profile",
            delta_color="inverse" if ttf < 24.0 else "normal"
        )
        
    with col2:
        st.metric(
            label="Prediction Reliability Confidence",
            value=f"{confidence:.1%}",
            delta="High Fidelity Model" if confidence >= 0.75 else "Low Density Telemetry Caution"
        )
        
    with col3:
        max_risk = max(node_risks.values()) if node_risks else 0.0
        status_label = "CRITICAL CONDITION" if max_risk > 0.70 else "WARNING DEPLETION" if max_risk > 0.40 else "OPTIMAL RUNTIME"
        st.metric(
            label="Peak Structural Node Risk",
            value=f"{max_risk:.2%}",
            delta=status_label,
            delta_color="inverse" if max_risk > 0.40 else "normal"
        )
        
    st.markdown("---")
    
    # ── HIGH PERFORMANCE BAR CHART COMPONENT ──
    st.write("### 🎛️ Distributed Subsystem Node Risk Vectors")
    
    if node_risks:
        risk_df = pd.DataFrame(list(node_risks.items()), columns=["Subsystem Component Node", "Risk Index Value"])
        # Enforce strict chart sorting for clear visual context
        risk_df = risk_df.sort_values(by="Risk Index Value", ascending=True)
        
        st.bar_chart(
            data=risk_df,
            x="Subsystem Component Node",
            y="Risk Index Value",
            use_container_width=True
        )
        
        with st.expander("👁️ Inspect Exact Component Values", expanded=False):
            st.table(risk_df.sort_values(by="Risk Index Value", ascending=False))
    else:
        st.info("No structural machine subsystem risk vector parameters present in current data state.")