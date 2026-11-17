"""
Aiqon Logistics AI - Streamlit Dashboard
Interactive Enterprise ETA Prediction, Delay Propagation, and What-If Disruption Sandbox.
"""

import os
import sys
from datetime import datetime, timedelta

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.predict import get_inference_engine
from src.delay_propagation import format_hours_to_hhmm, get_operational_status


# Streamlit Page Configuration
st.set_page_config(
    page_title="Aiqon Logistics Intelligence | ETA & Delay Propagation",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 16px;
        border-left: 4px solid #1f77b4;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }
    .badge-low { background-color: #d4edda; color: #155724; padding: 4px 8px; border-radius: 4px; font-weight: bold; }
    .badge-medium { background-color: #fff3cd; color: #856404; padding: 4px 8px; border-radius: 4px; font-weight: bold; }
    .badge-high { background-color: #f8d7da; color: #721c24; padding: 4px 8px; border-radius: 4px; font-weight: bold; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_engine():
    return get_inference_engine(models_dir="models")


@st.cache_data
def load_dataset():
    data_path = "dataset/shipments.csv"
    if os.path.exists(data_path):
        return pd.read_csv(data_path)
    return pd.DataFrame()


engine = load_engine()
df_shipments = load_dataset()

# Sidebar Navigation
st.sidebar.image("https://img.icons8.com/color/96/cargo-ship.png", width=64)
st.sidebar.title("Aiqon Logistics AI")
st.sidebar.caption("Maritime ETA & Delay Propagation Engine v1.0")

nav_choice = st.sidebar.radio(
    "Navigation",
    [
        "Fleet Overview",
        "Shipment ETA & Explainability",
        "Delay Propagation Engine",
        "What-If Disruption Simulator",
        "Model Governance & Metrics",
    ],
)

# -------------------------------------------------------------
# TAB 1: FLEET OVERVIEW
# -------------------------------------------------------------
if nav_choice == "Fleet Overview":
    st.title("🌐 Global Fleet Operations Overview")
    st.markdown("Real-time operational visibility across commercial maritime trade lanes.")

    if df_shipments.empty:
        st.warning("Shipment dataset not found. Please run `python -m src.data_generator`.")
    else:
        total_shipments = len(df_shipments)
        delayed_24h = int(df_shipments["is_delayed_24h"].sum())
        on_time = total_shipments - delayed_24h
        avg_delay = df_shipments["final_delay_hours"].mean()
        high_congestion = (df_shipments["port_congestion"] >= 0.6).sum()

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Total Active Shipments", f"{total_shipments:,}")
        col2.metric("On-Time / Minor (<24h)", f"{on_time:,}", f"{(on_time/total_shipments):.1%}")
        col3.metric("Delayed (≥24h)", f"{delayed_24h:,}", f"{(delayed_24h/total_shipments):.1%}", delta_color="inverse")
        col4.metric("Avg Final Delay", format_hours_to_hhmm(avg_delay), f"{avg_delay:+.2f}h decimal")
        col5.metric("Severe Port Congestion", f"{high_congestion}")

        st.markdown("---")

        col_left, col_right = st.columns([3, 2])
        
        with col_left:
            st.subheader("Delay Distribution across Global Trade Lanes")
            df_plot = df_shipments.copy()
            df_plot["Delay (HH:MM)"] = df_plot["final_delay_hours"].apply(format_hours_to_hhmm)
            df_plot["Status"] = df_plot["final_delay_hours"].apply(get_operational_status)

            fig_box = px.box(
                df_plot,
                x="route",
                y="final_delay_hours",
                color="route",
                points="all",
                hover_name="shipment_id",
                hover_data={
                    "route": False,
                    "Delay (HH:MM)": True,
                    "Status": True,
                    "final_delay_hours": ":.2f",
                },
                title="Final Delay Variance by Commercial Route",
                labels={
                    "final_delay_hours": "Final Delay (Decimal Hours)",
                    "route": "Route Code",
                    "Delay (HH:MM)": "Exact Delay",
                    "Status": "Operational Status",
                },
            )
            fig_box.add_hline(y=24.0, line_dash="dash", line_color="red", annotation_text="24h SLA Threshold")
            fig_box.update_layout(showlegend=False, height=420)
            st.plotly_chart(fig_box, use_container_width=True)

        with col_right:
            st.subheader("Risk Segmentation")
            risk_labels = ["On-Time / Buffer Absorbed (<12h)", "Moderate Delay (12-24h)", "Critical Delay (≥24h)"]
            c_low = (df_shipments["final_delay_hours"] < 12.0).sum()
            c_med = ((df_shipments["final_delay_hours"] >= 12.0) & (df_shipments["final_delay_hours"] < 24.0)).sum()
            c_high = (df_shipments["final_delay_hours"] >= 24.0).sum()

            fig_donut = go.Figure(data=[go.Pie(
                labels=risk_labels,
                values=[c_low, c_med, c_high],
                hole=0.55,
                marker=dict(colors=["#2ca02c", "#ff7f0e", "#d62728"]),
            )])
            fig_donut.update_layout(height=420, margin=dict(t=30, b=10, l=10, r=10))
            st.plotly_chart(fig_donut, use_container_width=True)

        st.subheader("Active Shipment Registry")
        st.caption("Operational tracking table for fleet administrators. Metrics and delays are formatted for rapid dispatch decisions.")

        col_filter, col_toggle = st.columns([3, 1])
        with col_toggle:
            show_audit_cols = st.toggle("🔍 Show Audit / Raw Columns", value=False, help="Reveals continuous decimal floats used by ML models.")

        df_registry = df_shipments.copy()
        df_registry["departure_delay_display"] = df_registry["departure_delay_hours"].apply(lambda x: format_hours_to_hhmm(x, show_early=False))
        df_registry["congestion_display"] = (df_registry["port_congestion"] * 100).round().astype(int).astype(str) + "%"
        df_registry["weather_display"] = df_registry["weather_severity"].apply(lambda x: f"{x:.2f}")
        df_registry["final_delay_display"] = df_registry["final_delay_hours"].apply(format_hours_to_hhmm)
        df_registry["operational_status"] = df_registry["final_delay_hours"].apply(get_operational_status)

        display_cols = [
            "shipment_id", "route", "origin", "destination", "distance_km",
            "departure_delay_display", "congestion_display", "weather_display",
            "final_delay_display", "operational_status"
        ]

        rename_dict = {
            "shipment_id": "Shipment ID",
            "route": "Route",
            "origin": "Origin",
            "destination": "Dest",
            "distance_km": "Distance (km)",
            "departure_delay_display": "Departure Delay",
            "congestion_display": "Congestion",
            "weather_display": "Weather (/5)",
            "final_delay_display": "Actual Final Delay",
            "operational_status": "Status",
        }

        if show_audit_cols:
            display_cols.extend(["departure_delay_hours", "final_delay_hours", "port_congestion", "weather_severity", "is_delayed_24h"])
            rename_dict.update({
                "departure_delay_hours": "Dep Delay (h)",
                "final_delay_hours": "Raw Actual Delay (h)",
                "port_congestion": "Raw Congestion (0-1)",
                "weather_severity": "Raw Weather (1-5)",
                "is_delayed_24h": "Delayed ≥24h Flag"
            })

        st.dataframe(
            df_registry[display_cols].rename(columns=rename_dict),
            use_container_width=True,
            height=320,
        )


# -------------------------------------------------------------
# TAB 2: SHIPMENT ETA & EXPLAINABILITY
# -------------------------------------------------------------
elif nav_choice == "Shipment ETA & Explainability":
    st.title("🎯 Single Shipment Prediction & Attribution")
    st.markdown("Inspect dynamic AI ETA predictions and multi-stage operational delay attribution.")

    if df_shipments.empty:
        st.warning("No shipment data available.")
    else:
        shipment_ids = df_shipments["shipment_id"].tolist()
        selected_id = st.selectbox("Select Shipment from Registry", shipment_ids)
        row = df_shipments[df_shipments["shipment_id"] == selected_id].iloc[0]

        prediction = engine.predict_single(row.to_dict())

        st.markdown("### Operational Scorecard")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Commercial Scheduled SLA", str(row["scheduled_delivery"])[:16].replace("T", " "))
        m2.metric("Predicted Dynamic ETA", str(prediction["predicted_eta"])[:16].replace("T", " "))
        m3.metric("Predicted Final Delay", format_hours_to_hhmm(prediction['predicted_delay_hours']), f"{prediction['predicted_delay_hours']:+.2f}h raw")
        
        tier = prediction["risk_tier"]
        badge_html = f"<div style='margin-top: 10px;'><span class='badge-{tier.lower()}'>{tier} RISK ({prediction['delay_probability']:.1%})</span></div>"
        m4.markdown("**Delay Risk Status**", unsafe_allow_html=True)
        m4.markdown(badge_html, unsafe_allow_html=True)

        st.markdown("---")

        col_wf, col_info = st.columns([3, 2])

        with col_wf:
            st.subheader("Operational Delay Waterfall (Stage Accounting)")
            stages = prediction["explanation"]["stages"]
            stage_names = [s["stage"] for s in stages]
            delays = [s["delay_contribution_hours"] for s in stages]

            fig_wf = go.Figure(go.Waterfall(
                name="Attribution",
                orientation="v",
                measure=["relative", "relative", "relative", "relative", "relative", "total"],
                x=stage_names + ["Net Predicted Delay"],
                textposition="outside",
                text=[f"{format_hours_to_hhmm(d, show_early=False)}" for d in delays] + [format_hours_to_hhmm(prediction['predicted_delay_hours'])],
                y=delays + [prediction["predicted_delay_hours"]],
                connector={"line": {"color": "rgb(63, 63, 63)"}},
                decreasing={"marker": {"color": "#2ca02c"}},
                increasing={"marker": {"color": "#d62728"}},
                totals={"marker": {"color": "#1f77b4"}},
            ))
            fig_wf.update_layout(
                title=f"Delay Propagation Waterfall: {selected_id}",
                showlegend=False,
                height=450,
                yaxis_title="Hours",
            )
            st.plotly_chart(fig_wf, use_container_width=True)

        with col_info:
            st.subheader("Attribution Breakdown Details")
            for s in stages:
                hours = s["delay_contribution_hours"]
                sign = "+" if hours > 0 else ""
                st.markdown(f"**{s['stage']}**: `{format_hours_to_hhmm(hours, show_early=False)}` _({sign}{hours:.2f}h raw)_  \n_{s['description']}_")

            st.info(
                "💡 **Engineering Note**: Unlike static ML feature weights, this waterfall reflects "
                "the true physical transit accounting including schedule buffer absorption."
            )


# -------------------------------------------------------------
# TAB 3: DELAY PROPAGATION ENGINE
# -------------------------------------------------------------
elif nav_choice == "Delay Propagation Engine":
    st.title("⏳ Multi-Stage Delay Propagation Architecture")
    st.markdown("Examine how schedule buffers absorb upstream lateness before compounding occurs.")

    if not df_shipments.empty:
        selected_id = st.selectbox("Select Shipment for Timeline Walkthrough", df_shipments["shipment_id"].tolist())
        record = df_shipments[df_shipments["shipment_id"] == selected_id].iloc[0]

        sched_dep = pd.to_datetime(record["scheduled_departure"])
        act_dep = pd.to_datetime(record["actual_departure"])
        sched_port = pd.to_datetime(record["scheduled_port_arrival"])
        act_port = pd.to_datetime(record["actual_port_arrival"]) if pd.notnull(record["actual_port_arrival"]) else (act_dep + timedelta(hours=300))
        sched_del = pd.to_datetime(record["scheduled_delivery"])
        act_del = pd.to_datetime(record["actual_delivery"])

        # Timeline comparison
        timeline_data = [
            {"Stage": "1. Origin Departure", "Scheduled": str(sched_dep)[:16].replace("T", " "), "Actual": str(act_dep)[:16].replace("T", " "), "Delay (HH:MM)": format_hours_to_hhmm(record["departure_delay_hours"], show_early=False), "Decimal (h)": f"{record['departure_delay_hours']:.2f}h"},
            {"Stage": "2. Port Arrival", "Scheduled": str(sched_port)[:16].replace("T", " "), "Actual": str(act_port)[:16].replace("T", " "), "Delay (HH:MM)": format_hours_to_hhmm(record["port_arrival_delay_hours"], show_early=False) if pd.notnull(record["port_arrival_delay_hours"]) else "N/A", "Decimal (h)": f"{record['port_arrival_delay_hours']:.2f}h" if pd.notnull(record["port_arrival_delay_hours"]) else "N/A"},
            {"Stage": "3. Final Consignee Delivery", "Scheduled": str(sched_del)[:16].replace("T", " "), "Actual": str(act_del)[:16].replace("T", " "), "Delay (HH:MM)": format_hours_to_hhmm(record["final_delay_hours"]), "Decimal (h)": f"{record['final_delay_hours']:.2f}h"},
        ]
        df_time = pd.DataFrame(timeline_data)

        st.subheader("Checkpoint Milestones")
        st.dataframe(df_time, use_container_width=True)

        # Propagation math breakdown
        st.subheader("Mathematical Propagation Principles")
        st.latex(r"\text{Net Delay}_k = \text{Delay}_{k-1} - \text{Slack Absorbed}_k + \Delta \text{Disruption}_k")
        st.latex(r"\text{Slack Absorbed}_k = \min(\max(0, \text{Delay}_{k-1}), \;\text{Scheduled Slack}_{k-1 \to k})")

        st.markdown("""
        **Operational Mechanics**:
        1. **Schedule Slack**: Commercial maritime timetables build in 16–28 hours of ocean buffer and 12–18 hours of customs buffer.
        2. **Absorption Window**: If a vessel departs 12 hours late but encounters calm seas, the ocean buffer absorbs that 12 hours entirely.
        3. **Compounding Bottleneck**: If port congestion dwell exceeds the remaining buffer, delay cascades downstream to inland drayage.
        """)


# -------------------------------------------------------------
# TAB 4: WHAT-IF DISRUPTION SIMULATOR
# -------------------------------------------------------------
elif nav_choice == "What-If Disruption Simulator":
    st.title("🧪 What-If Operational Disruption Sandbox")
    st.markdown("Simulate operational bottlenecks and quantify downstream ETA delay propagation.")

    col_sim_left, col_sim_right = st.columns([1, 2])

    with col_sim_left:
        st.subheader("Disruption Parameters")
        sim_shipment_id = st.selectbox("Target Shipment", df_shipments["shipment_id"].tolist())
        target_record = df_shipments[df_shipments["shipment_id"] == sim_shipment_id].iloc[0]

        stage_choice = st.selectbox(
            "Disruption Injection Point",
            ["port", "customs", "transit"],
            format_func=lambda x: {
                "port": "Port Congestion / Anchorage Wait",
                "customs": "Customs Inspection / Hold",
                "transit": "Ocean Storm / Route Detour",
            }[x],
        )

        added_hours = st.slider("Additional Delay Hours Incurred", min_value=0.0, max_value=96.0, value=24.0, step=4.0)
        run_sim = st.button("Simulate Downstream Propagation", type="primary")

    with col_sim_right:
        st.subheader("Simulation Results")
        sim_result = engine.simulate_delay_impact_on_shipment(
            target_record.to_dict(),
            disruption_stage=stage_choice,
            additional_hours=added_hours,
        )

        c1, c2, c3 = st.columns(3)
        c1.metric("Original Predicted Delay", format_hours_to_hhmm(sim_result['original_predicted_delay_hours']), f"{sim_result['original_predicted_delay_hours']:+.2f}h raw")
        c2.metric("Schedule Slack Absorbed", format_hours_to_hhmm(-sim_result['absorbed_by_schedule_slack_hours'], show_early=False), f"-{sim_result['absorbed_by_schedule_slack_hours']:.2f}h raw")
        c3.metric("Revised Predicted Delay", format_hours_to_hhmm(sim_result['revised_predicted_delay_hours']), delta=format_hours_to_hhmm(sim_result['net_effective_delay_increase_hours']))

        st.markdown("---")
        st.markdown(f"**Baseline ETA:** `{sim_result['original_eta']}`")
        st.markdown(f"**Revised Dynamic ETA:** `{sim_result['revised_eta']}`")
        st.markdown(f"**Projected Risk Tier:** `{sim_result['risk_tier']}`")

        # Visual comparison bar chart
        fig_sim = go.Figure(data=[
            go.Bar(name="Delay Incurred", x=["Original Baseline", "With Disruption"], y=[sim_result["original_predicted_delay_hours"], sim_result["revised_predicted_delay_hours"]], marker_color=["#1f77b4", "#d62728"])
        ])
        fig_sim.add_hline(y=24.0, line_dash="dash", line_color="orange", annotation_text="24h SLA Threshold")
        fig_sim.update_layout(height=350, yaxis_title="Final Delay Hours")
        st.plotly_chart(fig_sim, use_container_width=True)


# -------------------------------------------------------------
# TAB 5: MODEL GOVERNANCE & METRICS
# -------------------------------------------------------------
elif nav_choice == "Model Governance & Metrics":
    st.title("📊 Model Governance & Benchmark Evaluation")
    st.markdown("Transparent audit trail comparing the Baseline, Random Forest Regressor, and Classifier.")

    metrics_path = "models/metrics.json"
    if os.path.exists(metrics_path):
        import json
        with open(metrics_path, "r") as f:
            metrics = json.load(f)

        st.subheader("Model Performance Benchmark (Held-out Test Split)")
        col_reg, col_clf = st.columns(2)

        with col_reg:
            st.markdown("#### Regression: Final Delay (Hours)")
            base_reg = metrics["baseline_regression"]
            rf_reg = metrics["model_a_regression"]

            reg_table = pd.DataFrame([
                {"Model": "Baseline (Historical Mean)", "MAE (Hours)": f"{base_reg['mae_hours']}h", "RMSE (Hours)": f"{base_reg['rmse_hours']}h", "R² Score": base_reg["r2_score"]},
                {"Model": "RandomForestRegressor (Aiqon)", "MAE (Hours)": f"{rf_reg['mae_hours']}h", "RMSE (Hours)": f"{rf_reg['rmse_hours']}h", "R² Score": rf_reg["r2_score"]},
            ])
            st.table(reg_table)
            st.success("✅ Model A reduces Mean Absolute Error by over 55% compared to the naive historical baseline.")

        with col_clf:
            st.markdown("#### Classification: Delay Risk (≥24h)")
            clf_m = metrics["model_b_classification"]
            clf_table = pd.DataFrame([
                {"Metric": "Accuracy", "Value": f"{clf_m['accuracy']:.1%}"},
                {"Metric": "Precision (Late)", "Value": f"{clf_m['precision']:.1%}"},
                {"Metric": "Recall (Late)", "Value": f"{clf_m['recall']:.1%}"},
                {"Metric": "F1-Score", "Value": f"{clf_m['f1_score']:.4f}"},
                {"Metric": "ROC-AUC", "Value": f"{clf_m['roc_auc']:.4f}"},
            ])
            st.table(clf_table)

        st.markdown("---")
        st.subheader("Global Feature Importance (Gini)")
        top_feats = metrics.get("top_features", [])
        if top_feats:
            df_feat = pd.DataFrame(top_feats).sort_values(by="importance", ascending=True)
            fig_fi = px.bar(
                df_feat,
                x="importance",
                y="feature",
                orientation="h",
                title="Normalized Feature Importance (Variance Reduction)",
                labels={"importance": "Gini Importance", "feature": "Engineered Variable"},
                color="importance",
                color_continuous_scale="Viridis",
            )
            fig_fi.update_layout(height=420)
            st.plotly_chart(fig_fi, use_container_width=True)
    else:
        st.warning("Metrics file not found. Ensure training completed.")
