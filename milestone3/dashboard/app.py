"""
milestone3/dashboard/app.py

AeroDeep Multipage Streamlit Dashboard.
Main entry point — sets up navigation and shared state.

Run: streamlit run milestone3/dashboard/app.py
"""

import streamlit as st

st.set_page_config(
    page_title="AeroDeep — Fault Diagnostic System",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Shared session state defaults ─────────────────────────────────────────────
if "api_url" not in st.session_state:
    st.session_state.api_url = "http://localhost:8000"
if "selected_unit" not in st.session_state:
    st.session_state.selected_unit = "C1001A"
if "last_diagnostic" not in st.session_state:
    st.session_state.last_diagnostic = None
if "alert_history" not in st.session_state:
    st.session_state.alert_history = []

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://via.placeholder.com/200x60?text=AeroDeep", use_column_width=True)
    st.markdown("### Navigation")

    pages = {
        "🗺️  Asset Graph": "graph_view",
        "📊  Risk Monitor": "risk_panel",
        "🔬  Root Cause Diagnostics": "diagnostics",
        "⚙️  Settings": "settings",
    }

    for label, key in pages.items():
        if st.button(label, use_container_width=True):
            st.session_state.current_page = key

    st.divider()
    st.markdown("**Unit Selection**")
    units = ["C1001A", "C1001B", "C1002A", "K2001"]
    st.session_state.selected_unit = st.selectbox(
        "Active Unit", units,
        index=units.index(st.session_state.selected_unit),
    )

    st.divider()
    st.caption(f"API: {st.session_state.api_url}")
    st.caption("AeroDeep v1.0.0 | Gulf of Guinea Ops")


# ── Default landing page ──────────────────────────────────────────────────────
current = st.session_state.get("current_page", "graph_view")

if current == "graph_view":
    from milestone3.dashboard.graph_view import render_graph_view
    render_graph_view()
elif current == "risk_panel":
    from milestone3.dashboard.risk_panel import render_risk_panel
    render_risk_panel()
elif current == "diagnostics":
    from milestone3.dashboard.diagnostics import render_diagnostics
    render_diagnostics()
else:
    st.title("⚙️ Settings")
    st.text_input("API URL", value=st.session_state.api_url, key="api_url_input")
    if st.button("Save"):
        st.session_state.api_url = st.session_state.api_url_input
        st.success("Saved")
