"""
diagnostics.py
Dashboard Root Cause Analysis Panel. Processes classifications and maps asset failure probabilities.
"""
import streamlit as st
import pandas as pd

def render_diagnostics_panel(diagnostic: dict):
    """
    Renders sorted classification data tables and maps prescriptive mitigation
    checklists for onsite operators without risking dictionary key faults.
    """
    st.subheader("🔍 Continuous Root Cause Diagnostics")
    st.markdown("Deep contextual mapping vectors processed from system data models:")
    
    # Extract classification parameters using clean default structures
    faults = diagnostic.get("fault_probabilities", {}) if isinstance(diagnostic, dict) else {}
    top_faults = diagnostic.get("top_faults", []) if isinstance(diagnostic, dict) else []
    
    if faults:
        # Convert dictionary metrics into standard dataframes for visualization
        fault_df = pd.DataFrame(list(faults.items()), columns=["Diagnostic Failure Mode", "Probability Vector Value"])
        
        # Enforce strict descending sorting to prioritize high-risk anomalies
        fault_df = fault_df.sort_values(by="Probability Vector Value", ascending=False)
        
        st.dataframe(
            fault_df.style.format({"Probability Vector Value": "{:.2%}"})
            .background_gradient(cmap="Oranges", subset=["Probability Vector Value"]),
            use_container_width=True
        )
    else:
        st.info("No active failure probability distribution structures linked in this record frame.")

    # ── PRESCRIPTIVE MAINTENANCE CHEKLIST COMPONENT ──
    st.markdown("---")
    st.markdown("### 🚨 Urgent Prescriptive Mitigation Steps")
    
    if top_faults:
        st.caption("The classification engine generated targeted recovery tasks based on active fault limits:")
        
        for idx, item in enumerate(top_faults):
            fault_label = str(item[0])
            confidence_val = float(item[1])
            
            # Select appropriate threat boundaries and action plans based on urgency
            if confidence_val >= 0.70:
                alert_color = "red"
                action_text = "SCHEDULE IMMEDIATE MAINTENANCE: Shut down system loop, relieve internal pressures, and replace seals."
            elif confidence_val >= 0.40:
                alert_color = "orange"
                action_text = "INCREASE FIELD MONITORING: Inspect external wear profiles during the next shift change."
            else:
                alert_color = "blue"
                action_text = "LOG OPERATIONAL RUNTIME: Record telemetry signatures within standard baseline history logs."
                
            st.markdown(
                f"""
                <div style="border: 1px solid #374151; padding: 16px; margin-bottom: 12px; 
                            background-color: #111827; border-radius: 6px;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="font-weight: bold; color: #f9fafb; font-size: 1.05em;">Priority #{idx + 1}: {fault_label}</span>
                        <span style="background-color: {alert_color}; color: white; padding: 2px 8px; 
                                     border-radius: 12px; font-size: 0.8em; font-weight: bold;">
                            CONFIDENCE: {confidence_val:.1%}
                        </span>
                    </div>
                    <p style="color: #9ca3af; margin: 8px 0 0 0; font-size: 0.95em;">
                        <strong>Prescriptive Task:</strong> {action_text}
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )
    else:
        st.success("✅ **System Baseline Clear:** No active fault trends match current target metrics. Keep running standard operating profiles.")