# Industrial Predictive Maintenance & Intelligence Platform

**Digital Twin | Hybrid AI Diagnostics | Fleet Intelligence | Maintenance Planning | CP-SAT Optimization**

[![Live Demo](https://img.shields.io/badge/Live-Dashboard-FF4B4B?logo=streamlit&logoColor=white)](https://predictive-maintenance-ai-hjtlxffgfvdc7jd6dofuh5.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Framework-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Optimization](https://img.shields.io/badge/Optimization-OR--Tools-4285F4)](https://developers.google.com/optimization)

## Live Application

**Explore the deployed platform:**  
https://predictive-maintenance-ai-hjtlxffgfvdc7jd6dofuh5.streamlit.app/

## Overview

The Industrial Predictive Maintenance & Intelligence Platform is an end-to-end prototype that combines industrial machine simulation, AI-based fault diagnostics, fleet-level intelligence, maintenance planning, and mathematical optimization.

The platform simulates industrial equipment, injects controlled faults into sensor readings, evaluates diagnostic outputs using multiple AI models, generates maintenance work orders, and optimizes technician schedules using Google OR-Tools CP-SAT.

It also provides an interactive Streamlit dashboard for exploring machine health, fault diagnostics, maintenance operations, and optimization results.

## Platform Architecture

```text
Industrial Digital Twin
        |
        v
Multi-Machine Simulation
        |
        v
Sensor Readings & Fault Injection
        |
        v
Hybrid AI Diagnostic Engine
        |
        +---- XGBoost Failure Prediction
        +---- Isolation Forest Anomaly Detection
        +---- Fault Classification
        +---- Sensor Quality Assessment
        |
        v
Fleet Intelligence
        |
        +---- Fleet Health Analysis
        +---- Machine Risk Ranking
        +---- Fault Frequency Analysis
        |
        v
Maintenance Planning
        |
        +---- Work Order Generation
        +---- Spare Parts Forecasting
        +---- Technician Capacity Planning
        |
        v
OR-Tools CP-SAT Optimization
        |
        +---- Technician Assignment
        +---- Scheduling Constraints
        +---- Strategy Comparison
        |
        v
Interactive Streamlit Dashboard
```

## Key Features

### 1. Industrial Digital Twin

- Multi-machine industrial simulation.
- Configurable machine operating conditions.
- Simulated degradation and sensor readings.
- Controlled fault injection.
- Multiple machine operating scenarios.

### 2. Hybrid AI Diagnostics

- XGBoost-based machine failure prediction.
- Isolation Forest anomaly detection.
- Synthetic fault classification.
- Multi-label fault diagnostics.
- Sensor noise and quality assessment.
- Explainability using SHAP.

### 3. Fleet Intelligence

- Fleet-level health analysis.
- Machine severity classification.
- Rule-based machine risk ranking.
- Fault frequency analysis.
- Maintenance action recommendations.
- Sensor-quality fleet assessment.

### 4. Intelligent Maintenance Planning

- Automated maintenance work order generation.
- Fault-specific maintenance requirements.
- Spare parts forecasting.
- Technician capacity planning.
- Estimated maintenance cost and downtime.

### 5. CP-SAT Maintenance Optimization

Google OR-Tools CP-SAT is used to optimize maintenance scheduling.

The optimization framework considers:

- Technician assignment.
- Technician scheduling constraints.
- Prevention of overlapping jobs.
- Maintenance priorities.
- Job completion times.
- Scheduling strategy comparisons.

The platform compares five scheduling strategies:

1. Priority First
2. Shortest Processing Time
3. Severity First
4. FIFO
5. OR-Tools Optimal

### 6. Interactive Dashboard

The Streamlit application provides five pages:

- Executive Overview
- Fleet Intelligence
- AI Diagnostics
- Maintenance Center
- Optimization Lab

## Technology Stack

| Category | Technologies |
|---|---|
| Programming | Python |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn, XGBoost |
| Anomaly Detection | Isolation Forest |
| Explainable AI | SHAP |
| Optimization | Google OR-Tools CP-SAT |
| Visualization | Plotly |
| Dashboard | Streamlit |
| Testing | Python unittest |
| Version Control | Git, GitHub |

## Demonstrated Results

The current integrated simulation run produced the following results:

| Metric | Result |
|---|---:|
| Machines simulated | 8 |
| Simulation cycles | 100 |
| Diagnostic scenarios | 8 |
| Maintenance work orders | 7 |
| Maintenance scheduling days | 2 |
| Estimated downtime | 25 hours |
| Estimated maintenance cost | 2,870 simulation units |
| Optimization scenarios | 25 |
| Scheduling strategies compared | 5 |
| CP-SAT optimality proven | 25/25 scenarios |

**Optimization result:** OR-Tools CP-SAT achieved an average makespan of 12.50 hours across the 25 tested optimization scenarios.

These results describe the current controlled simulation experiments, not guaranteed performance in real industrial deployments.

## AI Model Evaluation

The XGBoost failure prediction model was trained and evaluated using the UCI AI4I 2020 Predictive Maintenance dataset.

Selected held-out test results:

| Metric | Score |
|---|---:|
| Precision | 0.8393 |
| Recall | 0.6912 |
| F1 Score | 0.7581 |
| ROC-AUC | 0.9784 |
| PR-AUC | 0.7493 |

The project also includes machine-held-out experiments for anomaly detection and synthetic fault classification.

**Important:** Results from controlled synthetic fault injection should not be interpreted as evidence of equivalent accuracy on real industrial equipment.

## Project Structure

```text
Predictive-Maintenance-AI/
│
├── app/
│   ├── dashboard.py
│   ├── components/
│   └── pages/
│
├── config/
│
├── data/
│
├── docs/
│
├── models/
│
├── notebooks/
│
├── reports/
│
├── screenshots/
│
├── src/
│   ├── simulation/
│   ├── intelligence/
│   ├── fleet_intelligence/
│   └── maintenance_planning/
│
├── tests/
│
├── run_platform.py
├── requirements.txt
└── README.md
```

## Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/vinayydv2025-del/Predictive-Maintenance-AI.git
cd Predictive-Maintenance-AI
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Generate the integrated platform report

```bash
python run_platform.py
```

### 5. Launch the dashboard

```bash
streamlit run app/dashboard.py
```

Open the local URL provided by Streamlit, usually:

```text
http://localhost:8501
```

## Run Tests

Execute the platform integration tests:

```bash
python -m unittest -v tests.test_platform_integration
```

The integration suite validates the platform pipeline, diagnostic and maintenance constraints, scheduling feasibility, and optimization scenario consistency.

## Data and Evaluation Limitations

This project is an experimental predictive maintenance prototype.

- The AI4I 2020 dataset is used for the XGBoost failure prediction model.
- The industrial digital twin and injected fault scenarios use simulated data.
- Synthetic fault classification results do not establish real-world industrial generalization.
- Machine risk ranking uses a transparent rule-based scoring system, not a calibrated failure probability.
- Maintenance costs and downtime are simulation estimates, not validated real-world values.
- CP-SAT optimality is established only for scenarios where the solver reports an optimal status.
- The platform is not a certified industrial safety or equipment-control system.

## Future Improvements

- Real-time IoT sensor integration.
- Remaining Useful Life (RUL) estimation with suitable real-world degradation data.
- FastAPI-based model serving.
- Docker-based deployment.
- Model monitoring and data drift detection.
- Automated model retraining.
- Edge deployment for industrial environments.
- Integration with real maintenance management systems.

## Author

**Vinay**

B.Tech — Information Technology

GitHub: https://github.com/vinayydv2025-del

---

**Live Demo:** https://predictive-maintenance-ai-hjtlxffgfvdc7jd6dofuh5.streamlit.app/

**Source Code:** https://github.com/vinayydv2025-del/Predictive-Maintenance-AI