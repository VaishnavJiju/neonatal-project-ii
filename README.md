# MAL-ED Clinical Nexus

> A full-stack clinical intelligence platform for neonatal health forecasting, built on the MAL-ED longitudinal cohort study.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![React](https://img.shields.io/badge/React-19-61DAFB?logo=react)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch)
![License](https://img.shields.io/badge/License-Academic-green)

---

## Overview

MAL-ED Clinical Nexus is an end-to-end machine learning platform that predicts child health outcomes using data from the **MAL-ED (Etiology, Risk Factors, and Interactions of Enteric Infections and Malnutrition and the Consequences for Child Health and Development)** multi-site birth cohort study spanning 8 countries.

The system combines **static tree-ensemble models** (XGBoost, CatBoost, Random Forest) with **deep sequence models** (LSTM, GRU, TCN) through a novel **hybrid architecture** that fuses learned temporal embeddings with clinical features for superior predictive performance.

### Prediction Targets

| Target | Type | Description |
|--------|------|-------------|
| **ΔBAZ** | Regression | Change in BMI-for-age Z-score between visits (growth velocity) |
| **Illness Burden** | Regression | Total illness days in the next ~30-day window |
| **Time-to-Recovery** | Regression | Days until first illness-free day (capped at 60) |
| **Diarrhea Risk** | Classification | Binary — will the child have diarrhea in the next window? |
| **BAZ Autoregressive** | Regression | Next visit's absolute Z-score |

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    React Frontend (Vite)                │
│  Explorer │ Laboratory │ Diagnostic │ Temporal Lab      │
│           │            │   Studio   │ + Simulation      │
└──────────────────────┬──────────────────────────────────┘
                       │ REST API
┌──────────────────────┴──────────────────────────────────┐
│                  FastAPI Backend                        │
│  Dataset API │ Feature Discovery │ Model Registry       │
│  Training    │ SHAP Engine       │ Copilot Proxy        │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────┴──────────────────────────────────┐
│               Model Registry (v0 / v1 / v2)            │
│  V0: Full-feature research    │ V2: Hybrid sequence     │
│  V1: Restricted clinical      │     (Embedding + XGB)   │
└─────────────────────────────────────────────────────────┘
```

### Three Model Tiers

- **V1 (Static)**: XGBoost/CatBoost/RF trained on ~11-12 curated clinical features per visit with GroupKFold CV
- **V2 (Sequence)**: LSTM, GRU, and TCN models trained on sliding-window temporal sequences
- **Hybrid**: LSTM/GRU/TCN embeddings extracted and fused with V1 clinical features → XGBoost for final prediction

---

## Platform Features

### 🔍 Data Explorer
- Dataset statistics (observations, features, completeness)
- Interactive univariate distribution analysis
- Searchable clinical codebook with 200+ feature definitions

### 🧪 Laboratory
- **Feature Discovery Engine** — Mutual Information, Correlation, SHAP, and Permutation importance with configurable sample sizes
- **Model Arena** — Train and compare multiple algorithms side-by-side with GroupKFold cross-validation, ablation controls, and diagnostic graphs

### 🩺 Diagnostic Studio
- **Explainability Suite** — Global SHAP importance, Beeswarm plots, Dependence plots, individual Force plots
- **Graph Explorer** — Prediction vs Actual, Residuals, Error Distribution (regression) or ROC, PR, Calibration, Threshold Analysis (classification)

### 🧬 Temporal Lab
- **Sequence Model Arena** — Train/evaluate LSTM, GRU, TCN models per target
- **Embedding Explorer** — PCA visualization of learned temporal representations
- **Simulation Engine** — What-if clinical scenario simulator: select a real child, adjust clinical parameters via sliders, and see real-time prediction changes with sensitivity trajectory curves

### 🤖 AI Clinical Copilot
- Context-aware AI assistant that watches your current view
- Powered by LLM (Groq API) with concise mode for quick clinical insights
- Dynamic suggestions based on active tab and graph type

---

## Project Structure

```
├── backend/
│   └── main.py                  # FastAPI server (all endpoints)
├── temporal_lab/
│   ├── build_sequence_datasets.py   # Sliding-window sequence tensors
│   ├── train_sequence_models.py     # LSTM/GRU/TCN training (PyTorch)
│   ├── extract_embeddings.py        # Hidden-state embedding extraction
│   ├── build_hybrid_datasets.py     # Merge embeddings + clinical features
│   ├── evaluate_sequence_models.py  # Sequence model evaluation
│   └── evaluate_all_paradigms.py    # V1 vs V2 vs Hybrid comparison
├── ui/src/
│   ├── App.jsx                  # Main app + Explorer tab
│   ├── Laboratory.jsx           # Feature Discovery + Model Arena
│   ├── DiagnosticStudio.jsx     # SHAP + Graph Explorer
│   ├── TemporalLab.jsx          # Sequence models + Simulation
│   ├── components/              # Reusable chart components
│   └── context/                 # React context providers
├── perfect_preprocess.py        # Raw data → master parquet
├── create_variants.py           # Lag/memory feature engineering
├── build_multi_target_pipeline.py   # 4 target datasets
├── build_v1_pipeline.py         # V1 clinical model training
├── registry_precompute.py       # Model registry builder
├── eval_multi_target.py         # Cross-target evaluation
├── run_ablation.py              # Ablation studies
├── stress_tests.py              # Model robustness tests
└── start.bat                    # One-click launcher
```

---

## Getting Started

### Prerequisites

- Python 3.10+
- Node.js 18+
- MAL-ED dataset files (not included — requires data access agreement)

### Installation

```bash
# Clone the repository
git clone https://github.com/VaishnavJiju/neonatal-project-ii.git
cd neonatal-project-ii

# Install Python dependencies
pip install fastapi uvicorn pandas numpy scikit-learn xgboost catboost lightgbm shap torch

# Install frontend dependencies
cd ui
npm install
cd ..
```

### Running the Platform

**Option 1: One-click launcher (Windows)**
```bash
start.bat
```

**Option 2: Manual**
```bash
# Terminal 1 — Backend
cd backend
python main.py
# → http://localhost:8000

# Terminal 2 — Frontend
cd ui
npm run dev
# → http://localhost:5173
```

### Pipeline Execution Order

To rebuild everything from raw data:

```
1. python perfect_preprocess.py              # Raw → master parquet
2. python create_variants.py                 # Feature engineering
3. python build_multi_target_pipeline.py     # Target datasets
4. python build_v1_pipeline.py               # V1 models + registry
5. cd temporal_lab
6. python build_sequence_datasets.py         # Sequence tensors
7. python train_sequence_models.py           # LSTM/GRU/TCN
8. python extract_embeddings.py              # Temporal embeddings
9. python build_hybrid_datasets.py           # Hybrid datasets
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | React 19, Vite, Recharts, Lucide Icons |
| Backend | FastAPI, Uvicorn |
| ML (Static) | scikit-learn, XGBoost, CatBoost, LightGBM |
| ML (Sequence) | PyTorch (LSTM, GRU, TCN) |
| Explainability | SHAP |
| AI Copilot | Groq API (LLM proxy) |
| Data | Pandas, Parquet |

---

## Data

This project uses the **MAL-ED 0-60 month** dataset. The raw data files (~1.4 GB) are **not included** in this repository due to size and licensing constraints. Data access can be requested from [ClinEpiDB](https://clinepidb.org/).

---

## Authors

Built as a final year academic project.

---

## License

This project is for academic and research purposes only. The MAL-ED dataset is subject to its own data use agreement.
