"""
data_manager.py
Centralized Data Normalization, Schema Integrity, and Ingestion Gateway Layer.
"""
from __future__ import annotations
import json
import pandas as pd
from typing import Dict, Any, Tuple, List

# Strict schema definitions required by presentation screens
EXPECTED_SCHEMA_KEYS = {
    "unit_id": str,
    "ttf_hours": (int, float),
    "ttf_confidence": (int, float),
    "node_risk": dict,
    "fault_probabilities": dict,
    "top_faults": list
}

def normalize_json_payload(raw_data: dict) -> Tuple[dict, List[str]]:
    """
    Validates and normalizes raw telemetry JSON input against the core diagnostic schema contract.
    Logs comprehensive alerts for schema deviations or missing parameters to prevent silent failures.
    """
    normalized = {}
    schema_warnings = []
    
    # 1. Asset identification validation
    normalized["unit_id"] = raw_data.get("unit_id")
    if normalized["unit_id"] is None:
        normalized["unit_id"] = "GENERIC_UNIT_00"
        schema_warnings.append("Missing payload key: 'unit_id'. Defaulted to 'GENERIC_UNIT_00'.")
    else:
        normalized["unit_id"] = str(normalized["unit_id"])
        
    # 2. Time-to-Failure threshold checking
    normalized["ttf_hours"] = raw_data.get("ttf_hours")
    if normalized["ttf_hours"] is None:
        normalized["ttf_hours"] = 168.0
        schema_warnings.append("Missing payload key: 'ttf_hours'. Enforced baseline standard of 168.0 hours.")
    elif not isinstance(normalized["ttf_hours"], (int, float)):
        try:
            normalized["ttf_hours"] = float(normalized["ttf_hours"])
        except (ValueError, TypeError):
            normalized["ttf_hours"] = 168.0
            schema_warnings.append("Malformed type for 'ttf_hours'. Coerced to baseline 168.0.")
            
    # 3. Model reliability score checking
    normalized["ttf_confidence"] = raw_data.get("ttf_confidence")
    if normalized["ttf_confidence"] is None:
        normalized["ttf_confidence"] = 0.50
        schema_warnings.append("Missing payload key: 'ttf_confidence'. Defaulted to 0.50 (50%).")
    elif not isinstance(normalized["ttf_confidence"], (int, float)):
        normalized["ttf_confidence"] = 0.50
        schema_warnings.append("Invalid data type for 'ttf_confidence'. Reset to fallback 0.50.")

    # 4. Spatial subsystem component risk map transformation
    raw_node_risk = raw_data.get("node_risk")
    if isinstance(raw_node_risk, dict):
        normalized["node_risk"] = {str(k): float(v) for k, v in raw_node_risk.items()}
    else:
        normalized["node_risk"] = {
            "hp_cylinder": 0.1, "lp_cylinder": 0.1, "intercooler": 0.1,
            "shaft_coupling": 0.1, "lube_oil_system": 0.1, "seal_system": 0.1
        }
        schema_warnings.append("Missing or malformed 'node_risk' map. Generated baseline nominal risks (0.10).")
        
    # 5. Fault probability arrays
    raw_faults = raw_data.get("fault_probabilities")
    if isinstance(raw_faults, dict):
        normalized["fault_probabilities"] = {str(k): float(v) for k, v in raw_faults.items()}
    else:
        normalized["fault_probabilities"] = {}
        schema_warnings.append("Missing 'fault_probabilities' dictionary object. Initialized empty state.")
        
    # 6. Prioritized anomaly tracking sequences
    raw_top_faults = raw_data.get("top_faults")
    if isinstance(raw_top_faults, list):
        normalized["top_faults"] = []
        for item in raw_top_faults:
            if isinstance(item, list) and len(item) >= 2:
                normalized["top_faults"].append([str(item[0]), float(item[1])])
            else:
                schema_warnings.append("Skipped improperly structured row item sequence inside 'top_faults' array.")
    else:
        normalized["top_faults"] = []
        schema_warnings.append("Missing 'top_faults' catalog sequence. Populated empty diagnostic list.")
        
    # Track extra variables to support downstream feature scaling
    for key, value in raw_data.items():
        if key not in normalized:
            normalized[key] = value
            
    return normalized, schema_warnings

def normalize_csv_tabular(df: pd.DataFrame) -> Tuple[dict, List[str]]:
    """
    Transforms structural column logs from an industrial spreadsheet (e.g., ai4i2020 layout)
    into the standardized dashboard dictionary payload format with active telemetry safety.
    """
    schema_warnings = []
    if df.empty:
        empty_payload, warnings = normalize_json_payload({})
        warnings.insert(0, "Target CSV spreadsheet dataframe is empty.")
        return empty_payload, warnings
        
    # Select the last active record row to represent real-time state
    target_row = df.iloc[-1]
    node_risk_map = {}
    
    if "Tool wear [min]" in df.columns:
        tool_wear_val = float(target_row.get("Tool wear [min]", 0))
        node_risk_map["hp_cylinder"] = min(tool_wear_val / 240.0, 1.0)
    else:
        schema_warnings.append("CSV Column 'Tool wear [min]' missing from spreadsheet layout. Filled default.")
        node_risk_map["hp_cylinder"] = 0.15
        
    # Derive structural stress alerts from machine binary exception flags
    hdf_triggered = int(target_row.get("HDF", 0)) == 1 if "HDF" in df.columns else False
    pwf_triggered = int(target_row.get("PWF", 0)) == 1 if "PWF" in df.columns else False
    twf_triggered = int(target_row.get("TWF", 0)) == 1 if "TWF" in df.columns else False
    osf_triggered = int(target_row.get("OSF", 0)) == 1 if "OSF" in df.columns else False
    
    node_risk_map["lp_cylinder"] = 0.55 if hdf_triggered else 0.12
    node_risk_map["lube_oil_system"] = 0.70 if pwf_triggered else 0.08
    node_risk_map["seal_system"] = 0.60 if osf_triggered else 0.10
    node_risk_map["intercooler"] = 0.05
    node_risk_map["shaft_coupling"] = 0.07
    
    failure_signal = False
    if "Machine failure" in df.columns:
        failure_signal = int(target_row.get("Machine failure", 0)) == 1
    else:
        schema_warnings.append("Critical alarm tracking column 'Machine failure' missing from file structure.")
        
    if failure_signal:
        calculated_ttf = 3.2
        confidence_score = 0.92
        top_faults_list = [["Active Industrial Machine Failure Warning Triggered", 1.0]]
    else:
        calculated_ttf = 134.5
        confidence_score = 0.68
        top_faults_list = []
        
    raw_payload = {
        "unit_id": str(target_row.get("Product ID", target_row.get("UDI", "TABULAR_ASSET_LOG"))),
        "ttf_hours": calculated_ttf,
        "ttf_confidence": confidence_score,
        "node_risk": node_risk_map,
        "fault_probabilities": {
            "Tool Wear Failure (TWF)": float(target_row.get("TWF", 0)) if "TWF" in df.columns else 0.0,
            "Heat Dissipation Failure (HDF)": float(target_row.get("HDF", 0)) if "HDF" in df.columns else 0.0,
            "Power Failure (PWF)": float(target_row.get("PWF", 0)) if "PWF" in df.columns else 0.0,
            "Overstrain Failure (OSF)": float(target_row.get("OSF", 0)) if "OSF" in df.columns else 0.0,
        },
        "top_faults": top_faults_list
    }
    
    return normalize_json_payload(raw_payload)