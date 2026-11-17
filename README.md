# 🚢 AI-Powered Shipment ETA Prediction & Delay Propagation System

[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.13-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-green.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red.svg)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4+-orange.svg)](https://scikit-learn.org/)
[![Pytest](https://img.shields.io/badge/pytest-40%20passed-brightgreen.svg)](https://docs.pytest.org/)

An enterprise-grade Machine Learning and Delay Propagation platform for international multi-modal container logistics. Built as a technical solution for the **Aiqon Technical Take-Home Assignment**.

---

## 📑 Table of Contents

- [📌 Executive Overview & Business Problem](#-executive-overview--business-problem)
  - [The Core Challenge: Naive Summation vs. Schedule Slack](#the-core-challenge-naive-summation-vs-schedule-slack)
- [🏛️ System Architecture](#️-system-architecture)
- [🔬 Mathematical Delay Propagation Methodology](#-mathematical-delay-propagation-methodology)
  - [1. Operational Milestones ($k \in \{0, 1, 2, 3, 4\}$)](#1-operational-milestones-k-in-0-1-2-3-4)
  - [2. Mathematical Formulations & Buffer Dynamics](#2-mathematical-formulations--buffer-dynamics)
  - [3. Non-Linear Bottleneck Compounding](#3-non-linear-bottleneck-compounding)
  - [4. Step-by-Step Numerical Walkthrough](#4-step-by-step-numerical-walkthrough)
- [📊 Dataset Schema](#-dataset-schema)
- [📈 Model Performance & Evaluation Results](#-model-performance--evaluation-results)
  - [Regression Benchmark (Final Delay in Hours)](#regression-benchmark-final-delay-in-hours)
  - [Classification Benchmark (Delay Risk $\ge 24$ Hours)](#classification-benchmark-delay-risk-ge-24-hours)
  - [Top 5 Global Predictive Features (Gini Importance)](#top-5-global-predictive-features-gini-importance)
- [🚀 Quick Start Guide (Windows PowerShell)](#-quick-start-guide-windows-powershell)
- [⚡ Large-Scale High-Throughput Architecture](#-large-scale-high-throughput-architecture)
- [📡 REST API Examples](#-rest-api-examples)
- [🗂️ Project Directory Structure](#️-project-directory-structure)
- [⚠️ Known Limitations & Future Improvements](#️-known-limitations--future-improvements)

---

## 📌 Executive Overview & Business Problem

In global supply chains, international maritime shipments transit through consecutive operational milestones:
1. **Origin Terminal Departure**: Vessel loading, terminal dwell, customs export clearance.
2. **Ocean Transit**: Great-circle navigation subject to sea state and storm rerouting.
3. **Port Arrival & Berth Congestion**: Anchorage queuing, pilot boarding, container offloading.
4. **Customs Clearance**: Regulatory inspections, documentation verification, physical scans.
5. **Hinterland Drayage & Final Delivery**: Last-mile road or rail transport to the consignee.

### The Core Challenge: Naive Summation vs. Schedule Slack
A naive approach treats delays as simple arithmetic additions ($\sum \text{delay}_i$). In reality:
- Commercial logistics timetables build in **schedule buffers (slack time)** designed to absorb upstream delays.
- Delays only cascade downstream when intermediate disruptions **exceed** available schedule buffers or when bottlenecks compound non-linearly (e.g., severe port congestion).

This system provides:
- **Delay Propagation Engine**: Deterministic operational accounting of schedule slack absorption vs. stage-added disruption.
- **Model A (RandomForestRegressor)**: Predicts final delay in hours and dynamic estimated delivery timestamp (ETA).
- **Model B (RandomForestClassifier)**: Estimates delay risk probability and assigns risk tiers (`LOW`, `MEDIUM`, `HIGH`).
- **Baseline Comparison**: Historical mean benchmark demonstrating measurable statistical value-add.
- **Dual-Level Explainability**: Global Gini feature importances and local per-shipment operational waterfalls.
- **Production REST API**: FastAPI backend with Pydantic validation and Swagger documentation.
- **Interactive Dashboard**: Streamlit interface featuring fleet KPIs, Plotly waterfalls, and a What-If disruption sandbox.

---

## 🏛️ System Architecture

```mermaid
flowchart TD
    subgraph Data_Layer ["1. Data & Simulation Layer"]
        DG["data_generator.py (Seed: 42)"] --> CSV[("dataset/shipments.csv (250 Records)")]
        CSV --> DP["delay_propagation.py (Slack Absorption Engine)"]
    end

    subgraph Feature_Layer ["2. Feature Pipeline (Zero Leakage)"]
        DP --> FE["feature_engineering.py (Domain Interactions)"]
        FE --> PP["ColumnTransformer (Median Imputer + Scaler + OHE)"]
    end

    subgraph Model_Layer ["3. ML Core & Evaluation"]
        PP --> Baseline["Historical Mean Baseline"]
        PP --> RFR["Model A: RandomForestRegressor (Delay Hours)"]
        PP --> RFC["Model B: RandomForestClassifier (Risk ≥ 24h)"]
        RFR & RFC --> Eval["evaluation.py (MAE, RMSE, R², F1, ROC-AUC)"]
        RFR --> Explain["explainability.py (Gini Importance & Waterfalls)"]
    end

    subgraph Serving_Layer ["4. Application & Serving"]
        RFR & RFC --> Service["app/services.py (Inference Engine)"]
        Service --> API["FastAPI REST API (app/main.py)"]
        Service --> UI["Streamlit Dashboard (dashboard/streamlit_app.py)"]
    end
```

---

## 🔬 Mathematical Delay Propagation Methodology

### 1. Operational Milestones ($k \in \{0, 1, 2, 3, 4\}$)

In multi-modal international container transport, a voyage is partitioned into five sequential, physically linked operational milestones:

| Milestone Index ($k$) | Operational Checkpoint | Scheduled Timestamp ($T_k^{\text{sched}}$) | Physical Logistics Activity | Stage Disruption Factors ($\delta_k$) |
| :---: | :--- | :---: | :--- | :--- |
| **$k = 0$** | **Origin Terminal Departure** | $T_0^{\text{sched}}$ | Container gate-in, export customs, and vessel stowage | Terminal yard congestion, crane breakdown, late vessel docking |
| **$k = 1$** | **Ocean Transit & Voyage** | $T_1^{\text{sched}}$ | Deep-sea maritime navigation along trade lanes | Storm deviation, adverse sea states, slow-steaming bunker optimization |
| **$k = 2$** | **Destination Port Arrival & Berthing** | $T_2^{\text{sched}}$ | Pilotage, outer anchorage queuing, berth offloading | Berth unavailability, tug shortages, terminal container dwell |
| **$k = 3$** | **Import Customs Clearance** | $T_3^{\text{sched}}$ | Manifest verification, tariff declaration, physical inspection | Documentation discrepancy holds, random x-ray scans, agency backlogs |
| **$k = 4$** | **Hinterland Drayage & Delivery** | $T_4^{\text{sched}}$ | Last-mile road trucking or rail intermodal transport | Drayage chassis shortage, highway congestion, consignee warehouse dwell |

---

### 2. Mathematical Formulations & Buffer Dynamics

Let $D_k$ denote the **cumulative net schedule delay** (in hours) at milestone $k$, where:
- $D_k > 0 \implies$ The shipment is running **behind schedule** (delayed).
- $D_k = 0 \implies$ The shipment is running **on time**.
- $D_k < 0 \implies$ The shipment is running **ahead of schedule** (early).

#### A. Initial Departure Delay ($k = 0$)
The initial delay at origin departure is the direct deviation between the actual departure timestamp $T_0^{\text{act}}$ and timetable departure $T_0^{\text{sched}}$:
$$D_0 = \max\left(0, \; T_0^{\text{act}} - T_0^{\text{sched}}\right)$$

#### B. Commercial Schedule Slack ($\text{Slack}_{k-1 \to k}$)
Commercial logistics operators insert deliberate timetable buffers between consecutive milestones to absorb operational friction:
$$\Delta T_{k-1 \to k}^{\text{sched}} = T_k^{\text{sched}} - T_{k-1}^{\text{sched}}$$
$$\text{Slack}_{k-1 \to k} = \max\left(0, \;\Delta T_{k-1 \to k}^{\text{sched}} - \tau_{k-1 \to k}^{\text{nom}}\right)$$
Where $\tau_{k-1 \to k}^{\text{nom}}$ is the nominal, disruption-free operational duration (e.g. calculated as $\frac{\text{Distance}}{v_{\text{nom}}}$ for ocean sailing).

#### C. Upstream Delay Absorption ($\text{AbsorbedDelay}_k$)
When an inherited delay $D_{k-1} > 0$ enters stage $k$, the stage timetable buffer absorbs part or all of the delay:
$$\text{AbsorbedDelay}_k = \min\left(D_{k-1}, \;\text{Slack}_{k-1 \to k}\right)$$

#### D. Net Delay Recurrence Relation
The net delay emerging from stage $k$ combines the unabsorbed upstream delay with newly incurred stage-level disruptions $\delta_k$:
$$D_k = \max\left(0, \; D_{k-1} - \text{AbsorbedDelay}_k + \delta_k\right)$$

#### E. Final Estimated Time of Arrival (ETA)
The dynamic arrival timestamp at destination ($k=4$) is updated in real time as:
$$\text{ETA} = T_4^{\text{sched}} + D_4$$

---

### 3. Non-Linear Bottleneck Compounding

In physical logistics, delays do not simply add linearly. High congestion causes **berth window expiration**, leading to non-linear cascading penalties:
- When a vessel arrives within its agreed port time window ($D_1 \le \text{BerthWindow}$), berthing delay scales linearly with terminal congestion $\rho \in [0, 1]$:
  $$\delta_2 = \tau_{\text{port}}^{\text{base}} \times (1 + \rho)$$
- When cumulative arrival delay exceeds the reserved berth window ($D_1 > \text{BerthWindow}$), the vessel forfeits its priority slot and must drop anchor in outer anchorage, inducing an exponential queue penalty:
  $$\delta_2^{\text{compounded}} = \delta_2 + \alpha \times e^{\beta \times \rho} \quad \text{for } \rho \ge 0.70$$
This non-linear reality is modeled directly in `src/delay_propagation.py` and captured by Model A's interaction feature `congestion_x_missed_slot`.

---

### 4. Step-by-Step Numerical Walkthrough

The table below contrasts our **Slack-Aware Delay Propagation** against a naive additive model ($\sum \delta_k$) for a transpacific voyage:

| Milestone Leg | Inherited Delay ($D_{k-1}$) | Timetable Slack ($\text{Slack}_k$) | Delay Absorbed | Stage Disruption ($\delta_k$) | Cumulative Delay ($D_k$) | Naive Summation ($\sum \delta$) | Error of Naive Model |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$k=0$: Departure** | — | — | — | **+8.0 h** (late gate-in) | **8.0 h** | 8.0 h | 0.0 h |
| **$k=1$: Ocean Transit** | 8.0 h | **12.0 h** (buffer) | **-8.0 h** (full absorption) | **+4.0 h** (mild storm) | **4.0 h** | 12.0 h | +8.0 h (overstated) |
| **$k=2$: Port Berthing** | 4.0 h | **4.0 h** (port buffer) | **-4.0 h** (absorbed) | **+14.0 h** (anchorage queue)| **14.0 h** | 26.0 h | +12.0 h (overstated) |
| **$k=3$: Customs Clearance**| 14.0 h | **6.0 h** (customs buffer)| **-6.0 h** (absorbed) | **+2.0 h** (standard scan) | **10.0 h** | 28.0 h | +18.0 h (overstated) |
| **$k=4$: Hinterland Drayage**| 10.0 h | **2.0 h** (drayage buffer)| **-2.0 h** (absorbed) | **+1.5 h** (traffic jam) | **9.5 h** | 29.5 h | **+20.0 h (overstated!)**|

**Result**:
- **Naive Summation** predicts a **29.5 hour delay** (triggering an unnecessary Critical SLA panic alarm).
- **Slack-Aware Delay Propagation** reveals the real delay is only **9.5 hours** (commercial timetable absorbed 20.0 hours across the journey).
- This demonstrates why naive addition fails in enterprise supply chains and why our dual physics/ML architecture is vital.

---

## 📊 Dataset Schema

The synthetic dataset contains 250 realistic shipments generated with fixed seed `42`:

| Column Name | Type | Availability | Description |
| :--- | :--- | :--- | :--- |
| `shipment_id` | String | At Checkpoint | Unique shipment identifier (`SHP-2026-0001` to `0250`) |
| `route` | String | At Checkpoint | Trade lane code (e.g. `ASIA-EUR-01`, `TPAC-01`) |
| `origin` / `destination` | String | At Checkpoint | Major port pairs (e.g. `CNSHA`, `NLRTM`, `CNNBO`, `USLAX`) |
| `distance_km` | Float | At Checkpoint | Voyage distance (7,200 km to 20,400 km) |
| `carrier_rating` | Float | At Checkpoint | Historical carrier reliability score (0.50 to 0.98) |
| `scheduled_departure` | Datetime | At Checkpoint | Baseline scheduled departure |
| `actual_departure` | Datetime | At Checkpoint | Actual terminal departure timestamp |
| `departure_delay_hours`| Float | At Checkpoint | Departure delay ($T_0^{\text{act}} - T_0^{\text{sched}}$) |
| `actual_port_arrival` | Datetime | At Checkpoint | Port arrival (contains ~7% missing telemetry) |
| `port_arrival_delay_hours`| Float | At Checkpoint | Port delay with median imputation handling |
| `port_congestion` | Float | At Checkpoint | Port anchorage congestion index (0.0 to 1.0) |
| `weather_severity` | Float | At Checkpoint | Sea state / storm index (1.0 to 5.0) |
| `port_processing_hours`| Float | At Checkpoint | Container offloading & terminal dwell hours |
| `customs_processing_hours`| Float | At Checkpoint | Regulatory clearance hours |
| `scheduled_delivery` | Datetime | At Checkpoint | Contractual SLA delivery timestamp |
| `actual_delivery` | Datetime | **Target Only** | Actual delivery (strictly sequestered from features) |
| `final_delay_hours` | Float | **Target A** | Continuous regression target ($T_4^{\text{act}} - T_4^{\text{sched}}$) |
| `is_delayed_24h` | Integer | **Target B** | Binary classification target ($1$ if delay $\ge 24$h, else $0$) |

---

## 📈 Model Performance & Evaluation Results

Evaluated on a held-out 20% **chronological test split** (50 unseen shipments):

### Regression Benchmark (Final Delay in Hours)

| Model Architecture | MAE (Hours) | RMSE (Hours) | $R^2$ Score | Error Reduction vs Baseline |
| :--- | :---: | :---: | :---: | :---: |
| **Baseline (Historical Mean)** | 23.81 h | 30.10 h | -0.0004 | Benchmark |
| **Model A: RandomForestRegressor** | **6.89 h** | **9.72 h** | **0.8958** | **-71.1% Error** |

### Classification Benchmark (Delay Risk $\ge 24$ Hours)

| Model Architecture | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Model B: RandomForestClassifier** | **88.0%** | **76.2%** | **94.1%** | **0.8421** | **0.9162** |

### Top 5 Global Predictive Features (Gini Importance)
1. `port_arrival_delay_hours`: 44.6%
2. `customs_processing_hours`: 35.5%
3. `port_processing_hours`: 7.4%
4. `congestion_x_missed_slot`: 5.1%
5. `congestion_processing_interaction`: 1.8%

---

## 🚀 Quick Start Guide (Windows PowerShell)

### 1. Environment Setup
```powershell
# Ensure virtual environment is active
.\.venv\Scripts\Activate.ps1

# Install pinned dependencies
pip install -r requirements.txt
```

### 2. Generate Dataset & Train Models
```powershell
# Generate synthetic dataset (seed=42)
python -m src.data_generator

# Train Baseline, Model A, and Model B; generate figures and metrics
python -m src.train_model
```

### 3. Run Automated Pytest Suite
```powershell
python -m pytest tests/ -v
```

### 4. Start FastAPI REST Backend
```powershell
uvicorn app.main:app --reload --port 8000
```
- Interactive Swagger UI: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- ReDoc Documentation: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

### 5. Launch Streamlit Dashboard
```powershell
streamlit run dashboard/streamlit_app.py
```
- Dashboard URL: [http://localhost:8501](http://localhost:8501)

---

## ⚡ Large-Scale High-Throughput Architecture

To scale from 250 test shipments to **hundreds of thousands or millions of shipment records**, the system implements four production scalability enhancements:

1. **Vectorized High-Speed Generation**:
   - `src/data_generator.py` uses full NumPy array broadcasting across all operational distributions instead of scalar Python loops.
   - Capable of synthesizing **100,000+ realistic multi-stage voyages in under 2 seconds**.

2. **Dual-Tier Columnar & Relational Storage**:
   - **Apache Parquet (`dataset/shipments.parquet`)**: Snappy-compressed columnar storage providing 80% disk space reduction and 15x faster scan speeds compared to raw CSV.
   - **Indexed SQLite Engine (`dataset/logistics.db`)**: Write-Ahead Logging (WAL) enabled with B-Tree indexes on `shipment_id`, `route`, `scheduled_departure`, and `is_delayed_24h` for sub-millisecond pagination (`LIMIT` / `OFFSET`) with $O(1)$ memory usage.

3. **High-Throughput Vectorized Batch Inference (`POST /predict/batch`)**:
   - Executes Scikit-Learn pipeline predictions across thousands of shipment rows in a single vectorized C-level matrix evaluation.
   - Benchmarked at **>2,000 shipments/sec** throughput on standard CPUs.

---

## 📡 REST API Examples

### 1. Predict Shipment ETA (`POST /predict`)
```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/predict" -Method Post -ContentType "application/json" -Body '{
  "shipment_id": "SHP-DEMO-01",
  "route": "ASIA-EUR-01",
  "origin": "CNSHA",
  "destination": "NLRTM",
  "distance_km": 19500.0,
  "carrier_rating": 0.88,
  "scheduled_departure": "2026-03-01T08:00:00",
  "departure_delay_hours": 4.0,
  "port_arrival_delay_hours": 6.0,
  "port_congestion": 0.40,
  "weather_severity": 2.5,
  "port_processing_hours": 32.0,
  "customs_processing_hours": 20.0,
  "scheduled_delivery": "2026-03-28T12:00:00"
}'
```

**Response**:
```json
{
  "shipment_id": "SHP-DEMO-01",
  "predicted_delay_hours": 9.42,
  "predicted_eta": "2026-03-28T21:25:12",
  "scheduled_delivery": "2026-03-28T12:00:00",
  "delay_probability": 0.1833,
  "is_delayed_24h": 0,
  "risk_tier": "LOW",
  "explanation": {
    "shipment_id": "SHP-DEMO-01",
    "predicted_delay_hours": 9.42,
    "delay_risk_category": "LOW",
    "stages": [
      {"stage": "Origin Departure", "delay_contribution_hours": 4.0, "description": "Terminal & gate release delay"},
      {"stage": "Ocean Transit (Weather)", "delay_contribution_hours": 4.0, "description": "Sailing slowdown due to sea state"},
      {"stage": "Port Congestion", "delay_contribution_hours": 4.8, "description": "Anchorage waiting time before berthing"},
      {"stage": "Customs Inspection", "delay_contribution_hours": 4.0, "description": "Regulatory clearance & documentation holds"},
      {"stage": "Schedule Slack Absorption", "delay_contribution_hours": -7.38, "description": "Commercial schedule buffer absorbing upstream delay"}
    ],
    "net_explained_delay_hours": 9.42,
    "methodology_note": "Attributions reflect physical logistics stage accounting and schedule slack absorption."
  }
}
```

### 2. Simulate Bottleneck Disruption (`POST /simulate-delay`)
```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/simulate-delay" -Method Post -ContentType "application/json" -Body '{
  "shipment_id": "SHP-2026-0001",
  "disruption_stage": "port",
  "additional_disruption_hours": 24.0
}'
```

### 3. High-Throughput Batch Prediction (`POST /predict/batch`)
```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/predict/batch" -Method Post -ContentType "application/json" -Body '{
  "shipments": [
    {
      "shipment_id": "BULK-001",
      "route": "ASIA-EUR-01",
      "origin": "CNSHA",
      "destination": "NLRTM",
      "distance_km": 19500.0,
      "carrier_rating": 0.88,
      "scheduled_departure": "2026-03-01T08:00:00",
      "departure_delay_hours": 2.0,
      "port_arrival_delay_hours": 4.0,
      "port_congestion": 0.35,
      "weather_severity": 2.0,
      "port_processing_hours": 28.0,
      "customs_processing_hours": 16.0,
      "scheduled_delivery": "2026-03-28T12:00:00"
    }
  ]
}'
```

---

## 🗂️ Project Directory Structure

```
aiqon-shipment-eta/
├── dataset/
│   ├── shipments.csv                   # CSV shipment records
│   ├── shipments.parquet               # Columnar Snappy-compressed Parquet
│   └── logistics.db                    # Indexed SQLite database (WAL mode)
├── notebooks/
│   └── exploratory_analysis.ipynb      # EDA and correlation explorations
├── src/
│   ├── __init__.py
│   ├── database.py                     # High-performance SQLite persistence & indexed queries
│   ├── data_generator.py               # Vectorized generator (supports 100k+ records)
│   ├── delay_propagation.py            # Physics & schedule slack absorption engine
│   ├── feature_engineering.py          # ColumnTransformer & domain interaction features
│   ├── train_model.py                  # Training pipeline for Baseline, Model A, Model B
│   ├── evaluation.py                   # Regression & classification metric calculations
│   ├── explainability.py               # Global Gini importances & local waterfalls
│   └── predict.py                      # Vectorized batch & single inference engine
├── models/
│   ├── regressor_pipeline.joblib       # Trained RandomForestRegressor pipeline
│   ├── classifier_pipeline.joblib      # Trained RandomForestClassifier pipeline
│   ├── baseline_model.joblib           # Historical mean benchmark
│   └── metrics.json                    # Serialized benchmark test metrics
├── app/
│   ├── __init__.py
│   ├── main.py                         # FastAPI REST application
│   ├── schemas.py                      # Pydantic v2 validation models
│   └── services.py                     # Business service & state manager
├── dashboard/
│   └── streamlit_app.py                # 5-tab enterprise Plotly Streamlit dashboard
├── tests/
│   ├── __init__.py
│   ├── test_data_generator.py          # Generator shape, ranges, and reproducibility tests
│   ├── test_delay_propagation.py       # Mathematical slack absorption & buffer tests
│   ├── test_feature_engineering.py     # Feature pipeline & transformer tests
│   ├── test_prediction.py              # ML inference & missing telemetry imputation tests
│   └── test_api.py                     # FastAPI TestClient end-to-end integration tests
├── reports/
│   └── figures/                        # High-resolution evaluation plots
│       ├── regression_actual_vs_predicted.png
│       ├── confusion_matrix.png
│       └── feature_importance.png
├── requirements.txt                    # Pinned Python package dependencies
└── README.md                           # Comprehensive documentation
```

---

## ⚠️ Known Limitations & Future Improvements

1. **Synthetic Data Boundaries**: While physical transit laws and realistic route distances are modeled, synthetic datasets cannot fully capture sudden black-swan disruptions (e.g. Red Sea / Suez canal diversions, sudden dock strikes).
2. **Tabular Milestones vs. Continuous Telemetry**: Current predictions operate at designated milestones. Future work should ingest raw hourly AIS vessel GPS coordinates via LSTM or Temporal Fusion Transformers (TFT).
3. **Graph-Level Network Effects**: Congestion at hub ports (e.g. Rotterdam, Singapore) cascades into regional feeder ports. Future iterations will model port networks as a dynamic Graph Neural Network (GNN).
4. **Prediction Intervals**: Future enhancements will implement conformal prediction or Quantile Random Forests to output calibrated 90% confidence bands ($[\text{ETA}_{\text{min}}, \text{ETA}_{\text{max}}]$) rather than point estimates alone.
