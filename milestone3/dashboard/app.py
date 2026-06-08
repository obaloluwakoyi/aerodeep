"""
milestone3/dashboard/app.py
Main Multi-Page Dashboard Shell, Routing Controller, and Gateway Interface.
"""
import os
import sys

# ── HIGH PRIORITY PATH INJECTION ──
# Forces Python to look locally FIRST, avoiding cloud environment naming collisions
local_dir = os.path.dirname(os.path.abspath(__file__))
if local_dir not in sys.path:
    sys.path.insert(0, local_dir)

import streamlit as st
import json
import pandas as pd

# Core architecture imports (Now guaranteed to look locally first)
from data_manager import normalize_json_payload, normalize_csv_tabular
from risk_panel import render_risk_panel
from graph_view import render_graph_view
from diagnostics import render_diagnostics_panel

st.set_page_config(
    page_title="AeroDeep Analytics Console",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── TRACK PERSISTENT STATE ACROSS NAVIGATION RERUNS ──
if "last_diagnostic" not in st.session_state:
    default_init, _ = normalize_json_payload({})
    st.session_state.last_diagnostic = default_init

if "schema_alerts" not in st.session_state:
    st.session_state.schema_alerts = []

# ── INGESTION GATEWAY SIDEBAR COMPONENT ──
st.sidebar.markdown("## 🚀 AeroDeep Ingestion Gateway")
st.sidebar.markdown("Upload raw files to run down-stream component views.")

uploaded_file = st.sidebar.file_uploader(
    "Upload Asset Diagnostic Feed", 
    type=["json", "csv"],
    help="Accepts streaming model JSON files or offline factory log rows."
)

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith(".json"):
            raw_content = json.load(uploaded_file)
            normalized, warnings = normalize_json_payload(raw_content)
        else:
            dataframe_content = pd.read_csv(uploaded_file)
            normalized, warnings = normalize_csv_tabular(dataframe_content)
            
        st.session_state.last_diagnostic = normalized
        st.session_state.schema_alerts = warnings
    except Exception as error:
        st.sidebar.error(f"Incompatible file structural format layout: {error}")

# ── COMPONENT TELEMETRY STREAM WARNING MONITOR ──
if st.session_state.schema_alerts:
    with st.sidebar.expander("⚠️ Pipeline Schema Alerts", expanded=True):
        st.caption("The normalization layer caught data anomalies and resolved them via fallbacks:")
        for alert in st.session_state.schema_alerts:
            st.markdown(f"- <small style='color:#f59e0b;'>{alert}</small>", unsafe_allow_html=True)
else:
    st.sidebar.success("✅ Architecture Schema Stream Synchronized")

# ── DASHBOARD SUB-PANEL ROUTING INTERFACE ──
st.sidebar.markdown("---")
st.sidebar.markdown("### 🧭 Sub-Panel Navigation")
selected_page = st.sidebar.radio(
    "Select Workspace View:",
    options=["📊 Risk Monitor Panel", "🕸️ Asset Topology Graph", "🔍 Root Cause Diagnostics"]
)

active_data = st.session_state.last_diagnostic

st.title("🛡️ AeroDeep Industrial Intelligence Console")
st.markdown(f"**Target Dynamic System Asset:** `{active_data.get('unit_id', 'Unknown')}`")
st.markdown("---")

# Execute isolated view module routing pipelines
if selected_page == "📊 Risk Monitor Panel":
    render_risk_panel(active_data)
elif selected_page == "🕸️ Asset Topology Graph":
    render_graph_view(active_data)
elif selected_page == "🔍 Root Cause Diagnostics":
    render_diagnostics_panel(active_data)