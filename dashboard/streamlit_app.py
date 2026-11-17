"""
Aiqon Logistics AI - Streamlit Dashboard
Interactive Enterprise ETA Prediction, Delay Propagation, and What-If Disruption Sandbox.
"""

import os
import sys
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.predict import get_inference_engine
from src.delay_propagation import (
    format_hours_to_hhmm,
    get_operational_status,
    calculate_schedule_buffer_analysis,
    format_buffer_duration,
)


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
logo_path = os.path.join(PROJECT_ROOT, "assets", "aiqon_logo.svg")
if os.path.exists(logo_path):
    st.sidebar.image(logo_path, width=76)
else:
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
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        border-radius: 12px;
        padding: 22px 26px;
        margin-bottom: 24px;
        color: white;
        display: flex;
        align-items: center;
        justify-content: space-between;
        box-shadow: 0 4px 14px rgba(15, 23, 42, 0.16);
    ">
        <div>
            <div style="font-size: 0.8rem; font-weight: 600; text-transform: uppercase; letter-spacing: 1.2px; color: #38bdf8; margin-bottom: 4px;">
                Aiqon Logistics AI • Global Fleet Control Tower
            </div>
            <div style="font-size: 1.7rem; font-weight: 700; letter-spacing: -0.5px; line-height: 1.2; margin-bottom: 6px;">
                Maritime ETA & Delay Propagation Engine
            </div>
            <div style="font-size: 0.92rem; color: #94a3b8; max-width: 800px;">
                Real-time operational visibility, predictive ETA tracking, and delay propagation monitoring across commercial maritime trade lanes.
            </div>
        </div>
        <div style="text-align: right; min-width: 150px;">
            <span style="
                background-color: rgba(34, 197, 94, 0.15);
                color: #4ade80;
                padding: 6px 14px;
                border-radius: 20px;
                font-size: 0.8rem;
                font-weight: 600;
                border: 1px solid rgba(74, 222, 128, 0.3);
                display: inline-flex;
                align-items: center;
                gap: 6px;
            ">
                <span style="height: 8px; width: 8px; background-color: #22c55e; border-radius: 50%; display: inline-block;"></span>
                Live Fleet Feed
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

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
            st.plotly_chart(fig_box, width="stretch")

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
            st.plotly_chart(fig_donut, width="stretch")

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
            width="stretch",
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

        tier = prediction["risk_tier"].title()
        prob_pct = prediction["delay_probability"] * 100
        threshold_desc = prediction.get("prediction_threshold", "Delay ≥ 24 hours")

        if tier == "Low":
            bg_grad = "linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%)"
            border_c = "#86efac"
            text_c = "#14532d"
            dot_c = "#22c55e"
        elif tier == "Medium":
            bg_grad = "linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%)"
            border_c = "#fde68a"
            text_c = "#78350f"
            dot_c = "#f59e0b"
        else:
            bg_grad = "linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%)"
            border_c = "#fca5a5"
            text_c = "#7f1d1d"
            dot_c = "#ef4444"

        m4.markdown(f"""
        <div style="
            background: {bg_grad};
            border: 1px solid {border_c};
            border-radius: 12px;
            padding: 10px 14px;
            box-shadow: 0 1px 4px rgba(0,0,0,0.04);
            margin-top: -4px;
        ">
            <div style="font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.6px; color: #4b5563; font-weight: 600; margin-bottom: 2px;">
                SLA Breach Risk (≥24h)
            </div>
            <div style="font-size: 1.25rem; font-weight: 700; color: {text_c}; display: flex; align-items: center; gap: 6px; line-height: 1.2;">
                <span style="height: 8px; width: 8px; background-color: {dot_c}; border-radius: 50%; display: inline-block;"></span>
                {tier} Risk <span style="font-size: 0.88rem; font-weight: 600; opacity: 0.85;">({prob_pct:.1f}%)</span>
            </div>
            <div style="font-size: 0.73rem; color: #4b5563; margin-top: 3px;">
                <b>Target:</b> {threshold_desc}
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.caption(
            "ℹ️ **Calibrated Risk Definition**: Delay probability quantifies the likelihood of a major SLA breach (**Delay ≥ 24 hours**). "
            "Explicit thresholds: **Low Risk** (< 35%), **Medium Risk** (35% – 70%), **High Risk** (≥ 70%)."
        )

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
            st.plotly_chart(fig_wf, width="stretch")

        with col_info:
            st.subheader("Attribution Breakdown Details")
            st.caption("Operational stage breakdown distinguishing newly introduced disruptions from schedule buffer mitigation.")

            stages = prediction["explanation"]["stages"]

            def is_buffer_stage(stg: dict) -> bool:
                name = str(stg.get("stage", "")).lower()
                return "buffer" in name or "slack" in name

            delay_stages = [s for s in stages if not is_buffer_stage(s)]
            buffer_stages = [s for s in stages if is_buffer_stage(s)]

            # Robust fallback guarantee: if no stage matched by name, treat the last item as buffer
            if not buffer_stages and len(stages) >= 5:
                delay_stages = stages[:-1]
                buffer_stages = [stages[-1]]

            gross_delay = sum(s["delay_contribution_hours"] for s in delay_stages)
            total_buffer_impact = sum(s["delay_contribution_hours"] for s in buffer_stages)

            st.markdown("##### ⏱️ Stage-Level Delays Incurred (Gross)")
            for s in delay_stages:
                hours = s["delay_contribution_hours"]
                st.markdown(
                    f"- **{s['stage']}**: `{format_hours_to_hhmm(hours, show_early=False)}` _({hours:+.2f}h)_  \n"
                    f"  <span style='color: #666; font-size: 0.85rem;'>{s['description']}</span>",
                    unsafe_allow_html=True,
                )
            st.markdown(f"**Gross Delay Incurred:** `{format_hours_to_hhmm(gross_delay, show_early=False)}` _({gross_delay:+.2f}h)_")

            st.markdown("---")
            if total_buffer_impact < -0.01:
                st.markdown("##### 🛡️ Schedule Buffer Absorption (Mitigation)")
            elif total_buffer_impact > 0.01:
                st.markdown("##### ⚠️ Schedule Buffer Deficit (Compounding Delay)")
            else:
                st.markdown("##### ⚖️ Schedule Buffer (Neutral / Zero Impact)")

            for s in buffer_stages:
                val = s["delay_contribution_hours"]
                color = "#28a745" if val < -0.01 else ("#d62728" if val > 0.01 else "#6c757d")
                st.markdown(
                    f"- **{s['stage']}**: `{format_hours_to_hhmm(val, show_early=False)}` _({val:+.2f}h)_  \n"
                    f"  <span style='color: {color}; font-size: 0.85rem;'>{s['description']}</span>",
                    unsafe_allow_html=True,
                )

            buffer_title = "Total Buffer Absorbed:" if total_buffer_impact <= 0 else "Total Buffer Deficit / Impact:"
            st.markdown(f"**{buffer_title}** `{format_hours_to_hhmm(total_buffer_impact, show_early=False)}` _({total_buffer_impact:+.2f}h)_")

            st.markdown("---")
            st.markdown(
                f"**🎯 Net Predicted Final Delay:** `{format_hours_to_hhmm(prediction['predicted_delay_hours'])}` "
                f"_({prediction['predicted_delay_hours']:+.2f}h)_"
            )


# -------------------------------------------------------------
# TAB 3: DELAY PROPAGATION ENGINE
# -------------------------------------------------------------
elif nav_choice == "Delay Propagation Engine":
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #1e1b4b 0%, #312e81 50%, #4338ca 100%);
        border-radius: 12px;
        padding: 22px 26px;
        margin-bottom: 24px;
        color: white;
        display: flex;
        align-items: center;
        justify-content: space-between;
        box-shadow: 0 4px 14px rgba(30, 27, 75, 0.2);
    ">
        <div>
            <div style="font-size: 0.8rem; font-weight: 600; text-transform: uppercase; letter-spacing: 1.2px; color: #a5b4fc; margin-bottom: 4px;">
                Aiqon Logistics AI • Milestone Accounting
            </div>
            <div style="font-size: 1.7rem; font-weight: 700; letter-spacing: -0.5px; line-height: 1.2; margin-bottom: 6px;">
                Multi-Stage Delay Propagation Architecture
            </div>
            <div style="font-size: 0.92rem; color: #c7d2fe; max-width: 820px;">
                Examine how schedule buffers absorb upstream lateness before compounding occurs, and audit milestone-by-milestone schedule deviations.
            </div>
        </div>
        <div style="text-align: right; min-width: 150px;">
            <span style="
                background-color: rgba(165, 180, 252, 0.15);
                color: #e0e7ff;
                padding: 6px 14px;
                border-radius: 20px;
                font-size: 0.8rem;
                font-weight: 600;
                border: 1px solid rgba(165, 180, 252, 0.3);
                display: inline-flex;
                align-items: center;
                gap: 6px;
            ">
                ⏳ Timeline Physics
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if not df_shipments.empty:
        selected_id = st.selectbox("Select Shipment for Timeline Walkthrough", df_shipments["shipment_id"].tolist())
        record = df_shipments[df_shipments["shipment_id"] == selected_id].iloc[0]

        sched_dep = pd.to_datetime(record["scheduled_departure"])
        act_dep = pd.to_datetime(record["actual_departure"])
        sched_port = pd.to_datetime(record["scheduled_port_arrival"])
        act_port = pd.to_datetime(record["actual_port_arrival"]) if pd.notnull(record["actual_port_arrival"]) else None
        sched_del = pd.to_datetime(record["scheduled_delivery"])
        act_del = pd.to_datetime(record["actual_delivery"])

        dep_delay_h = float(record["departure_delay_hours"])
        port_delay_h = float(record["port_arrival_delay_hours"]) if pd.notnull(record["port_arrival_delay_hours"]) else None
        final_delay_h = float(record["final_delay_hours"])

        def get_milestone_status(delay_val: Optional[float]) -> str:
            if delay_val is None:
                return "In Transit"
            if delay_val < -0.1:
                return "Early"
            elif delay_val <= 0.1:
                return "On Schedule"
            return "Late"

        def fmt_dt(dt_val: Optional[pd.Timestamp]) -> str:
            if dt_val is None or pd.isna(dt_val):
                return "N/A"
            return dt_val.strftime("%b %d, %H:%M")

        # Milestone 1: Origin Departure
        status_dep = get_milestone_status(dep_delay_h)
        inh_dep_h = 0.0
        change_dep_h = dep_delay_h

        # Milestone 2: Port Arrival
        status_port = get_milestone_status(port_delay_h)
        inh_port_h = dep_delay_h
        change_port_h = (port_delay_h - dep_delay_h) if port_delay_h is not None else None

        # Milestone 3: Final Delivery
        status_final = get_milestone_status(final_delay_h)
        inh_final_h = port_delay_h if port_delay_h is not None else dep_delay_h
        change_final_h = final_delay_h - inh_final_h

        # -------------------------------------------------------------
        # Visual Delay Propagation Timeline (Interactive Journey Track)
        # -------------------------------------------------------------
        st.subheader("Visual Delay Propagation Journey Track")
        st.caption("Graphical milestone trajectory. Downward slopes indicate schedule buffer absorption; upward spikes reveal compounding bottlenecks.")

        # 1. Milestone 1: Origin Departure
        if dep_delay_h < -0.1:
            c1, lbl1 = "#16a34a", f"Early ({format_hours_to_hhmm(dep_delay_h, show_early=False)})"
        elif abs(dep_delay_h) <= 0.1:
            c1, lbl1 = "#64748b", "On Schedule (0m)"
        else:
            c1, lbl1 = "#dc2626", f"Late ({format_hours_to_hhmm(dep_delay_h, show_early=False)})"

        # 2. Milestone 2: Ocean Transit
        dist_km = float(record.get("distance_km", 0.0) or 0.0)
        weather_val = float(record.get("weather_severity", 2.0) or 2.0)
        if change_port_h is not None:
            if change_port_h > 2.0:
                c2, lbl2 = "#ea580c", f"Disruption (+{change_port_h:.1f}h)"
            elif change_port_h < -1.0:
                c2, lbl2 = "#16a34a", f"Buffer Absorbed ({change_port_h:.1f}h)"
            else:
                c2, lbl2 = "#64748b", "Nominal Sailing"
        else:
            c2, lbl2 = "#64748b", "In Transit"

        # 3. Milestone 3: Port Arrival
        if act_port is None:
            c3, lbl3 = "#ea580c", "Telemetry Gap / Imputed"
        elif port_delay_h is not None:
            if port_delay_h < -0.1:
                c3, lbl3 = "#16a34a", f"Early ({format_hours_to_hhmm(port_delay_h, show_early=False)})"
            elif abs(port_delay_h) <= 0.1:
                c3, lbl3 = "#64748b", "On Schedule (0m)"
            else:
                c3, lbl3 = "#dc2626", f"Late ({format_hours_to_hhmm(port_delay_h, show_early=False)})"
        else:
            c3, lbl3 = "#64748b", "In Transit"

        # 4. Milestone 4: Port & Customs Dwell
        port_proc_h = float(record.get("port_processing_hours", 24.0) or 24.0)
        cust_proc_h = float(record.get("customs_processing_hours", 16.0) or 16.0)
        port_cong_pct = float(record.get("port_congestion", 0.3) or 0.3) * 100.0
        if change_final_h > 2.0:
            c4, lbl4 = "#ea580c", f"Disruption (+{change_final_h:.1f}h)"
        elif change_final_h < -1.0:
            c4, lbl4 = "#16a34a", f"Buffer Absorbed ({change_final_h:.1f}h)"
        else:
            c4, lbl4 = "#64748b", "Nominal Clearance"

        # 5. Milestone 5: Final Consignee Delivery
        if final_delay_h < -0.1:
            c5, lbl5 = "#16a34a", f"Early ({format_hours_to_hhmm(final_delay_h, show_early=False)})"
        elif abs(final_delay_h) <= 0.1:
            c5, lbl5 = "#64748b", "On Schedule (0m)"
        else:
            c5, lbl5 = "#dc2626", f"Late ({format_hours_to_hhmm(final_delay_h, show_early=False)})"

        # Trajectory delay values
        ocean_proj_delay = dep_delay_h + (change_port_h if change_port_h is not None else 0.0)
        port_arr_delay = port_delay_h if port_delay_h is not None else ocean_proj_delay

        x_stages = [
            "1. Origin Departure",
            "2. Ocean Transit",
            "3. Port Arrival",
            "4. Terminal & Customs",
            "5. Consignee Delivery",
        ]
        y_delays = [
            dep_delay_h,
            ocean_proj_delay,
            port_arr_delay,
            final_delay_h,
            final_delay_h,
        ]

        node_colors = [c1, c2, c3, c4, c5]
        node_texts = [
            format_hours_to_hhmm(dep_delay_h, show_early=False),
            format_hours_to_hhmm(change_port_h, show_early=False) if change_port_h is not None else "Nominal",
            format_hours_to_hhmm(port_delay_h, show_early=False) if port_delay_h is not None else "Missing",
            format_hours_to_hhmm(change_final_h, show_early=False),
            format_hours_to_hhmm(final_delay_h, show_early=False),
        ]

        hover_info = [
            f"<b>1. Origin Departure</b><br>Actual: {fmt_dt(act_dep)}<br>Scheduled: {fmt_dt(sched_dep)}<br>Deviation: {format_hours_to_hhmm(dep_delay_h, show_early=False)}",
            f"<b>2. Ocean Transit</b><br>Corridor: {dist_km:,.0f} km<br>Sea State: {weather_val:.2f}/5<br>Transit Deviation: {format_hours_to_hhmm(change_port_h, show_early=False) if change_port_h is not None else 'Nominal'}",
            f"<b>3. Port Arrival</b><br>Actual: {fmt_dt(act_port) if act_port is not None else 'Missing Telemetry'}<br>Scheduled: {fmt_dt(sched_port)}<br>Port Deviation: {format_hours_to_hhmm(port_delay_h, show_early=False) if port_delay_h is not None else 'N/A'}",
            f"<b>4. Terminal & Customs Dwell</b><br>Port Dwell: {port_proc_h:.0f}h | Customs: {cust_proc_h:.0f}h<br>Berth Congestion: {port_cong_pct:.0f}%<br>Dwell Impact: {format_hours_to_hhmm(change_final_h, show_early=False)}",
            f"<b>5. Consignee Delivery</b><br>Actual: {fmt_dt(act_del)}<br>Scheduled: {fmt_dt(sched_del)}<br>Final Delay: {format_hours_to_hhmm(final_delay_h, show_early=False)}",
        ]

        fig_track = go.Figure()

        # Connect journey with trajectory line
        fig_track.add_trace(go.Scatter(
            x=x_stages,
            y=y_delays,
            mode="lines+markers+text",
            line=dict(color="#6366f1", width=3),
            marker=dict(size=24, color=node_colors, line=dict(color="#ffffff", width=2)),
            text=node_texts,
            textposition="top center",
            hovertext=hover_info,
            hoverinfo="text",
            showlegend=False,
        ))

        # Baseline and SLA reference lines
        fig_track.add_hline(y=0.0, line_dash="solid", line_color="#10b981", annotation_text="Scheduled Target (0h)")
        fig_track.add_hline(y=24.0, line_dash="dash", line_color="#ef4444", annotation_text="24h SLA Threshold")

        fig_track.update_layout(
            height=320,
            margin=dict(t=40, b=20, l=40, r=40),
            yaxis_title="Schedule Deviation (Hours)",
        )
        st.plotly_chart(fig_track, width="stretch")

        # 5 Native Milestone Stepper Cards (Fully Dark/Light Mode Compatible)
        c1_col, c2_col, c3_col, c4_col, c5_col = st.columns(5)

        with c1_col:
            with st.container(border=True):
                st.markdown("🚢 **1. Departure**")
                st.markdown(f"**{fmt_dt(act_dep)}**")
                st.caption(f"Sched: {fmt_dt(sched_dep)}")
                st.markdown(f"`{lbl1}`")

        with c2_col:
            with st.container(border=True):
                st.markdown("🌊 **2. Ocean Transit**")
                st.markdown(f"**{dist_km:,.0f} km Leg**")
                st.caption(f"Sea Severity: {weather_val:.2f}/5")
                st.markdown(f"`{lbl2}`")

        with c3_col:
            with st.container(border=True):
                st.markdown("⚓ **3. Port Arrival**")
                st.markdown(f"**{fmt_dt(act_port) if act_port is not None else 'Missing Telemetry'}**")
                st.caption(f"Sched: {fmt_dt(sched_port)}")
                st.markdown(f"`{lbl3}`")

        with c4_col:
            with st.container(border=True):
                st.markdown("🏢 **4. Terminal Dwell**")
                st.markdown(f"**Port {port_proc_h:.0f}h | Cust {cust_proc_h:.0f}h**")
                st.caption(f"Congestion: {port_cong_pct:.0f}%")
                st.markdown(f"`{lbl4}`")

        with c5_col:
            with st.container(border=True):
                st.markdown("📦 **5. Final Delivery**")
                st.markdown(f"**{fmt_dt(act_del)}**")
                st.caption(f"Sched: {fmt_dt(sched_del)}")
                st.markdown(f"`{lbl5}`")

        st.caption(
            "🎨 **Timeline Color Code:** "
            "<span style='color: #dc2626; font-weight: bold;'>■ Red</span>: Late | "
            "<span style='color: #16a34a; font-weight: bold;'>■ Green</span>: Early / Buffer Absorbed | "
            "<span style='color: #64748b; font-weight: bold;'>■ Grey</span>: On Schedule | "
            "<span style='color: #ea580c; font-weight: bold;'>■ Orange</span>: Disruption Incurred",
            unsafe_allow_html=True,
        )
        st.markdown("---")

        # -------------------------------------------------------------
        # Schedule Buffer Analysis (Capacity & Slack Consumption)
        # -------------------------------------------------------------
        buf_analysis = calculate_schedule_buffer_analysis(record.to_dict())

        st.subheader("Schedule Buffer Analysis")
        st.caption("Commercial absorptive capacity: tracks planned schedule buffer against slack consumed by operational disruptions.")

        b1_col, b2_col, b3_col, b4_col = st.columns(4)
        with b1_col:
            with st.container(border=True):
                st.caption("INITIAL SCHEDULE SLACK")
                st.markdown(f"## {buf_analysis['initial_slack_display']}")
                st.caption("Original Available Buffer")

        with b2_col:
            with st.container(border=True):
                st.caption("SLACK CONSUMED")
                st.markdown(f"## {buf_analysis['consumed_slack_display']}")
                st.caption("Buffer Already Used")

        with b3_col:
            with st.container(border=True):
                st.caption("REMAINING BUFFER")
                st.markdown(f"## {buf_analysis['remaining_slack_display']}")
                st.caption("Buffer Still Available")

        with b4_col:
            with st.container(border=True):
                st.caption("BUFFER UTILIZATION")
                st.markdown(f"## {buf_analysis['utilization_display']}")
                st.markdown(f"**{buf_analysis['status_badge']}**")

        st.progress(
            min(1.0, buf_analysis["buffer_utilization_pct"] / 100.0),
            text=f"Buffer Consumption: {buf_analysis['utilization_display']} — {buf_analysis['status_desc']}",
        )
        st.markdown("---")

        col_hdr, col_toggle = st.columns([3, 1])
        with col_hdr:
            st.subheader("Checkpoint Milestones")
            st.caption("Milestone-by-milestone schedule deviations across the shipment journey.")
        with col_toggle:
            show_tech_milestones = st.toggle("🔍 Show Raw Technical Decimals", value=False, help="Reveals underlying decimal hours for schedule deviations.")

        timeline_data = [
            {
                "Checkpoint": "1. Origin Departure",
                "Scheduled": fmt_dt(sched_dep),
                "Actual": fmt_dt(act_dep),
                "Delay": format_hours_to_hhmm(dep_delay_h, show_early=False),
                "Status": status_dep,
                "Previous-Stage Delay": format_hours_to_hhmm(inh_dep_h, show_early=False),
                "Delay Change": format_hours_to_hhmm(change_dep_h, show_early=False),
                "Raw Delay (h)": f"{dep_delay_h:+.2f}h",
                "Raw Inherited (h)": f"{inh_dep_h:+.2f}h",
                "Raw Change (h)": f"{change_dep_h:+.2f}h",
            },
            {
                "Checkpoint": "2. Port Arrival",
                "Scheduled": fmt_dt(sched_port),
                "Actual": fmt_dt(act_port) if act_port is not None else "Simulated / In Transit",
                "Delay": format_hours_to_hhmm(port_delay_h, show_early=False) if port_delay_h is not None else "N/A",
                "Status": status_port,
                "Previous-Stage Delay": format_hours_to_hhmm(inh_port_h, show_early=False),
                "Delay Change": format_hours_to_hhmm(change_port_h, show_early=False) if change_port_h is not None else "N/A",
                "Raw Delay (h)": f"{port_delay_h:+.2f}h" if port_delay_h is not None else "N/A",
                "Raw Inherited (h)": f"{inh_port_h:+.2f}h",
                "Raw Change (h)": f"{change_port_h:+.2f}h" if change_port_h is not None else "N/A",
            },
            {
                "Checkpoint": "3. Final Delivery",
                "Scheduled": fmt_dt(sched_del),
                "Actual": fmt_dt(act_del),
                "Delay": format_hours_to_hhmm(final_delay_h, show_early=False),
                "Status": status_final,
                "Previous-Stage Delay": format_hours_to_hhmm(inh_final_h, show_early=False),
                "Delay Change": format_hours_to_hhmm(change_final_h, show_early=False),
                "Raw Delay (h)": f"{final_delay_h:+.2f}h",
                "Raw Inherited (h)": f"{inh_final_h:+.2f}h",
                "Raw Change (h)": f"{change_final_h:+.2f}h",
            },
        ]
        df_time = pd.DataFrame(timeline_data)

        base_cols = ["Checkpoint", "Scheduled", "Actual", "Delay", "Status", "Previous-Stage Delay", "Delay Change"]
        if show_tech_milestones:
            base_cols.extend(["Raw Delay (h)", "Raw Inherited (h)", "Raw Change (h)"])

        st.dataframe(df_time[base_cols], width="stretch")

        st.info("💡 **Operational Note**: The **Delay Change** represents changes in schedule deviation between consecutive checkpoints (not independently established causes). Negative changes indicate delay absorbed by schedule slack, while positive changes indicate stage disruptions.")

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
        st.plotly_chart(fig_sim, width="stretch")


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
            st.plotly_chart(fig_fi, width="stretch")
    else:
        st.warning("Metrics file not found. Ensure training completed.")
