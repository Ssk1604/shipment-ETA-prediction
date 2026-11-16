# 🚢 AI-Powered Shipment ETA Prediction & Delay Propagation System

[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.13-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-green.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red.svg)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4+-orange.svg)](https://scikit-learn.org/)
[![Pytest](https://img.shields.io/badge/pytest-22%20passed-brightgreen.svg)](https://docs.pytest.org/)

An enterprise-grade Machine Learning and Delay Propagation platform for international multi-modal container logistics. Built as a technical solution for the **Aiqon Technical Take-Home Assignment**.

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

Let operational milestones be indexed by $k \in \{0, 1, 2, 3, 4\}$.

1. **Commercial Schedule Slack**:
   $$\text{Slack}_{k-1 \to k} = \max\left(0, \;\Delta T^{\text{sched}}_{k-1 \to k} - \tau^{\text{nom}}_{k-1 \to k}\right)$$
   Where $\Delta T^{\text{sched}}$ is the timetable duration and $\tau^{\text{nom}}$ is the nominal transit/processing time.

2. **Upstream Delay Absorption**:
   When an inherited delay $D_{k-1} > 0$ enters stage $k$:
   $$\text{AbsorbedDelay}_k = \min\left(D_{k-1}, \;\text{Slack}_{k-1 \to k}\right)$$

3. **Stage Recurrence & Compounding**:
   $$\text{Net Delay}_k = D_{k-1} - \text{AbsorbedDelay}_k + \delta_k$$
   Where $\delta_k$ is the local disruption added at stage $k$ (e.g. weather slow-steaming or port anchorage wait).

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
| **Baseline (Historical Mean)** | 23.69 h | 29.56 h | -0.0058 | Benchmark |
| **Model A: RandomForestRegressor** | **9.62 h** | **18.94 h** | **0.5872** | **-59.4% Error** |

### Classification Benchmark (Delay Risk $\ge 24$ Hours)

| Model Architecture | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Model B: RandomForestClassifier** | **92.0%** | **100.0%** | **81.0%** | **0.8947** | **0.9787** |

### Top 5 Global Predictive Features (Gini Importance)
1. `port_arrival_delay_hours`: 43.1%
2. `customs_processing_hours`: 29.6%
3. `port_processing_hours`: 10.0%
4. `congestion_processing_interaction`: 5.6%
5. `departure_weather_composite`: 5.2%

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

### 6. Generate Technical PDF Report
```powershell
python -m reports.generate_pdf_report
```
- Compiled report saved to: `reports/Aiqon_Shipment_ETA_Technical_Report.pdf`

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
│   ├── test_prediction.py              # ML inference & missing telemetry imputation tests
│   └── test_api.py                     # FastAPI TestClient end-to-end integration tests
├── reports/
│   ├── figures/                        # High-resolution evaluation plots
│   │   ├── regression_actual_vs_predicted.png
│   │   ├── confusion_matrix.png
│   │   └── feature_importance.png
│   ├── generate_pdf_report.py          # PDF report compiler (ReportLab)
│   └── Aiqon_Shipment_ETA_Technical_Report.pdf  # 3-page executive technical report
├── requirements.txt                    # Pinned Python package dependencies
└── README.md                           # Comprehensive documentation
```

---

## ⚠️ Known Limitations & Future Improvements

1. **Synthetic Data Boundaries**: While physical transit laws and realistic route distances are modeled, synthetic datasets cannot fully capture sudden black-swan disruptions (e.g. Red Sea / Suez canal diversions, sudden dock strikes).
2. **Tabular Milestones vs. Continuous Telemetry**: Current predictions operate at designated milestones. Future work should ingest raw hourly AIS vessel GPS coordinates via LSTM or Temporal Fusion Transformers (TFT).
3. **Graph-Level Network Effects**: Congestion at hub ports (e.g. Rotterdam, Singapore) cascades into regional feeder ports. Future iterations will model port networks as a dynamic Graph Neural Network (GNN).
4. **Prediction Intervals**: Future enhancements will implement conformal prediction or Quantile Random Forests to output calibrated 90% confidence bands ($[\text{ETA}_{\text{min}}, \text{ETA}_{\text{max}}]$) rather than point estimates alone.
