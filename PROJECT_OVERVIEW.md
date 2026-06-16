# MAL-ED Clinical Nexus: Project Overview

## 🧠 Project Vision

The MAL-ED Clinical Nexus is a full-stack clinical forecasting and diagnostic intelligence platform for pediatric health outcomes. It predicts **child growth (BAZ)**, **growth velocity (ΔBAZ)**, **diarrhea risk**, **illness burden**, and **recovery time** using the MAL-ED 0–60 month multi-site longitudinal cohort (1,532,191 observations, 1,793 unique children, 176 clinical features).

The core innovation is a **three-tier modeling paradigm**:
1. **Static tabular models** (V1) — restricted, interpretable clinical models using ~11–12 features.
2. **Deep temporal sequence models** — LSTM and TCN architectures trained on windowed patient histories.
3. **Hybrid fusion models** — LSTM/TCN embeddings concatenated with clinical features, boosted by XGBoost.

All models, predictions, and explanations are served through a **Registry-Driven Architecture** backed by a modern React + FastAPI stack with an embedded AI Copilot for clinical decision support.

---

## 🏗️ Architecture: V0 → V1 → V2

### V0: Research / Full-Feature Model
- **Scope**: Uses all 171 available features (SES, survey data, clinical markers).
- **Purpose**: Provides a "theoretical upper bound" on performance.
- **Risk**: Prone to "cheating" via survey proxies (e.g., using wealth indices to predict nutrition).

### V1: Clinical / Restricted Static Model
- **Scope**: Uses a strict whitelist of ~11–12 clinical markers per target (age, weight, recent illness history, antibiotics, WAMI index).
- **Algorithms**: CatBoost, XGBoost, Random Forest — trained with GroupKFold (by PID) cross-validation.
- **Purpose**: Designed for real-world clinical deployment where only bedside medical history is available.
- **Registry**: All artifacts precomputed and stored in `models/v1/{target}/{algorithm}/`.

### V2: Temporal Sequence & Hybrid Models
- **Scope**: Uses windowed temporal sequences of the same clinical features.
- **Architectures**: LSTM, TCN (Temporal Convolutional Network), and Hybrid (LSTM/TCN embeddings + XGBoost).
- **Purpose**: Captures longitudinal temporal patterns (e.g., a child who has been declining for 3 consecutive visits).
- **Registry**: Stored in `models/v2/` with `full_paradigm_results.json` and `prediction_samples.json`.

---

## 📊 Model Performance Summary

### Static Models (V1) — All 4 Targets

#### BAZ Forecast (Autoregressive)
| Model | R² | MAE | RMSE |
|---|---|---|---|
| **CatBoost** | **0.843** | **0.298** | **0.448** |
| XGBoost | 0.838 | 0.302 | 0.456 |
| Random Forest | 0.838 | 0.306 | 0.456 |

#### Growth Velocity (ΔBAZ)
| Model | R² | MAE | RMSE |
|---|---|---|---|
| **CatBoost** | **0.079** | **0.293** | **0.444** |
| Random Forest | 0.057 | 0.299 | 0.449 |
| XGBoost | 0.042 | 0.299 | 0.453 |

#### Diarrhea Risk (Classification)
| Model | ROC-AUC | F1 | F2 |
|---|---|---|---|
| **CatBoost** | **0.873** | 0.830 | 0.787 |
| XGBoost | 0.866 | 0.923 | 0.933 |
| Random Forest | 0.847 | 0.921 | 0.932 |

#### Illness Burden (Regression)
| Model | R² | MAE | RMSE |
|---|---|---|---|
| **CatBoost** | **0.640** | **2.110** | **4.314** |
| XGBoost | 0.629 | 2.116 | 4.378 |
| Random Forest | 0.626 | 2.144 | 4.396 |

> **Champion Static Model**: CatBoost leads across all targets on R² and MAE.

### Temporal & Hybrid Models (V2) — 3 Regression Targets

#### Growth Velocity (ΔBAZ)
| Model | R² | MAE | RMSE |
|---|---|---|---|
| Baseline (Zero Growth) | -0.002 | 0.285 | 0.430 |
| V1 (XGBoost Tabular) | 0.126 | 0.273 | 0.402 |
| LSTM | 0.155 | 0.267 | 0.395 |
| TCN | 0.153 | 0.268 | 0.396 |
| **Hybrid LSTM** | **0.546** | **0.188** | **0.290** |
| Hybrid TCN | 0.475 | 0.206 | 0.311 |

#### Illness Burden
| Model | R² | MAE | RMSE |
|---|---|---|---|
| Baseline (Persistence) | 0.128 | 2.149 | 5.377 |
| V1 (XGBoost Tabular) | 0.530 | 1.675 | 3.947 |
| LSTM | 0.538 | 1.701 | 3.917 |
| TCN | 0.529 | 1.768 | 3.953 |
| **Hybrid LSTM** | **0.609** | **1.487** | **3.601** |
| Hybrid TCN | 0.602 | 1.543 | 3.631 |

#### Recovery (Time-to-Recovery from Acute Malnutrition)
| Model | R² | MAE (days) | RMSE (days) |
|---|---|---|---|
| Baseline (Last Interval) | -2.160 | 31.310 | 79.621 |
| V1 (XGBoost Tabular) | 0.516 | 12.947 | 31.171 |
| LSTM | 0.512 | 12.075 | 31.297 |
| TCN | 0.489 | 13.638 | 32.012 |
| **Hybrid LSTM** | **0.511** | **11.687** | **31.320** |
| Hybrid TCN | 0.455 | 12.938 | 33.070 |

> **Champion Hybrid Model**: Hybrid LSTM (LSTM embeddings + XGBoost) wins across all temporal targets on MAE, with the most dramatic improvement on ΔBAZ (MAE: 0.273 → 0.188, a 31% reduction).

---

## 📂 Project Structure

```
FINAL PROJECT DATASET/
│
├── backend/
│   └── main.py                      # FastAPI backend (67K lines) — all API endpoints
│
├── ui/                              # React + Vite frontend
│   └── src/
│       ├── App.jsx                  # Main app shell, routing, Intake & Explorer tab
│       ├── Laboratory.jsx           # Feature Discovery + Model Arena
│       ├── DiagnosticStudio.jsx     # SHAP Suite + Diagnostic Graph Explorer
│       ├── TemporalLab.jsx          # Sequence Modeling + Simulation Engine
│       ├── NexusContext.jsx          # Global state (version, target, model)
│       ├── index.css                # Global design system
│       └── components/
│           ├── CopilotWidget.jsx    # AI Copilot (resizable, context-aware)
│           ├── ClassificationGraphs.jsx  # ROC, PR, Calibration curves
│           ├── RegressionGraphs.jsx      # Scatter, Residuals, Error Dist.
│           ├── Beeswarm.jsx         # SHAP Beeswarm visualization
│           ├── ForcePlot.jsx        # Per-patient SHAP Force Plot
│           ├── Dependence.jsx       # SHAP Dependence Plots
│           └── SHAPImportance.jsx   # Feature Importance bar charts
│
├── models/
│   ├── v1/                          # Static model registry
│   │   ├── manifest.json
│   │   ├── baz_ar/{catboost,random_forest,xgboost}/
│   │   ├── delta_baz/{catboost,random_forest,xgboost}/
│   │   ├── diarrhea/{catboost,random_forest,xgboost}/
│   │   └── illness_burden/{catboost,random_forest,xgboost}/
│   │       └── Each: metadata.json, predictions.parquet, shap_global.json,
│   │               shap_sample.parquet, expected_value.json
│   │
│   └── v2/                          # Temporal model registry
│       ├── full_paradigm_results.json
│       ├── prediction_samples.json
│       ├── lstm_*.pt, tcn_*.pt       # Trained PyTorch sequence models
│       ├── *_hybrid_lstm_xgb.pkl     # Hybrid model pickles
│       └── scaler_*.pkl              # Feature scalers
│
├── temporal_lab/                    # Temporal model training pipeline
│   ├── build_sequence_datasets.py   # Windowed dataset creation
│   ├── train_sequence_models.py     # LSTM/TCN training
│   ├── extract_embeddings.py        # Latent embedding extraction
│   ├── build_hybrid_datasets.py     # Embedding + clinical merge
│   ├── evaluate_sequence_models.py  # Individual model evaluation
│   └── evaluate_all_paradigms.py    # Full benchmark comparison
│
├── mal_ed_data/                     # Source parquet datasets
│   ├── dataset_baz_ar.parquet
│   ├── dataset_delta_baz.parquet
│   ├── dataset_diarrhea.parquet
│   ├── dataset_illness_burden.parquet
│   └── enriched_codebook.csv        # Clinical variable definitions (165 entries)
│
├── images/                          # Report-ready charts (dark theme)
├── images_light/                    # Report-ready charts (light theme)
│   ├── static_models/
│   │   ├── comparison_tables/       # Tabular comparison images
│   │   └── catboost_results/        # Scatter plots + SHAP importance
│   ├── temporal_models/
│   │   ├── comparison_tables/       # Temporal benchmark tables
│   │   └── best_hybrid_results/     # Hybrid LSTM scatter plots
│   └── trajectories/               # Longitudinal growth curves
│
├── registry_precompute_v1.py        # V1 registry generation pipeline
├── build_v1_pipeline.py             # Clinical feature whitelist definition
├── build_multi_target_pipeline.py   # Multi-target pipeline builder
├── stress_tests.py                  # Automated leakage/stress testing
└── PROJECT_OVERVIEW.md              # This file
```

---

## 🖥️ Frontend: The MAL-ED Nexus Dashboard

### Design Philosophy
- **Glassmorphism dark UI** with vibrant cyan accents, frosted glass panels, and micro-animations.
- **Five main tabs**: Explorer → Laboratory → Diagnostic → Temporal Lab → Copilot.
- **Version toggle** (Clinical V1 / Full V0) persisted globally via `NexusContext`.

### Tab 1: Intake & Explorer
- **Data Preview & Metrics**: Shows 1,532,191 patients, 176 features, 1,793 children, 98.6% data completeness.
- **Live Tensor Feed**: Scrollable sample table of raw patient observations.
- **Distribution Analysis**: Interactive histograms for any selected feature.
- **Clinical Codebook**: Searchable reference of all 165 research variables with clinical definitions and domain tags.

### Tab 2: The Laboratory
- **Feature Discovery**: Correlation analysis with selectable target. Runs Mutual Information, Correlation, SHAP, and Permutation importance methods.
- **Model Arena**: Real-time model training and comparison.
  - **Training Configuration**: Target selection, version toggle, task type (regression/classification), model version.
  - **Registry Mode**: Instantly loads precomputed results from the registry.
  - **Force Retrain Mode**: Live training with feature ablation support.
  - **Model Diagnostic Graphs**: Scatter plots, confusion matrices, and feature importance via `POST /api/models/visualize`.

### Tab 3: Diagnostic Studio
- **SHAP Suite**: Beeswarm plots, global importance bars, SHAP dependence plots — all registry-driven.
- **Force Plot Explorer**: Per-patient SHAP force plots showing which features push risk up vs. down.
- **Classification Graphs** (for diarrhea target): ROC curves, Precision-Recall curves, Calibration plots, and threshold analysis.
- **Regression Graphs** (for regression targets): Predicted vs. Actual scatter, residual analysis, error distribution.

### Tab 4: Temporal Intelligence Lab
- **Sequence Modeling**: Multi-stage pipeline simulator.
  - Step 1: Train temporal backbone (LSTM or TCN).
  - Step 2: Extract latent embeddings.
  - Step 3: Merge with clinical features to create hybrid dataset.
  - Step 4: Benchmark hybrid model against all baselines.
  - Graph options: R² Validation, Residuals, Error Distribution.
- **Simulation Engine**: What-if patient simulator.
  - Select a real child from the cohort.
  - Adjust clinical features (age, weight, illness burden) via sliders.
  - See real-time model prediction changes.

### Tab 5 / Floating Widget: Nexus Copilot
- **AI-powered clinical assistant** using OpenAI GPT-4.
- **Context-aware**: Automatically receives the active tab, target, model, graph type, metrics, and graph data.
- **Resizable**: Draggable resize handle (min 320×400, max 800×900).
- **Smart prompts**: Suggested questions like "Explain this view" and "What drives changes in growth velocity?"
- **Clinical reasoning**: System prompt enforces multi-metric comparison rules:
  - Regression: MAE > RMSE > R² priority order.
  - Classification: F1/F2 > Recall > AUC > Precision.
  - Must flag metric disagreements and never declare a winner on a single metric.

---

## ⚙️ Backend: FastAPI (`backend/main.py`)

### Core Components
- **`RegistryManager`**: Centralized class for all model artifact lookups. Methods:
  - `get_metadata()`, `get_predictions()`, `get_shap_global()`, `get_shap_sample()`
  - `get_expected_value()`, `get_classification_curves()`, `get_threshold_metrics()`
- **`GLOBAL_STATE`**: In-memory dataset store loaded at startup via `PROJECT_ROOT` resolution.
- **Feature whitelists**: Per-target clinical feature subsets enforced during training.

### Key API Endpoints
| Endpoint | Method | Purpose |
|---|---|---|
| `/api/data/info` | GET | Dataset summary (rows, columns, children) |
| `/api/data/sample` | GET | Live tensor feed (paginated sample) |
| `/api/data/distribution/{col}` | GET | Histogram data for any feature |
| `/api/data/dictionary` | GET | Clinical codebook entries |
| `/api/models/score_target` | POST | Train and score models (force retrain) |
| `/api/models/visualize` | POST | Scatter/confusion matrix + feature importance |
| `/api/models/registry/*` | GET | All registry artifact endpoints |
| `/api/models/feature_discovery` | POST | Multi-method feature importance analysis |
| `/api/temporal/metrics` | GET | V2 benchmark results |
| `/api/temporal/embeddings` | GET | Extracted temporal embeddings |
| `/api/temporal/simulate/*` | GET/POST | What-if simulation engine |
| `/api/copilot/chat` | POST | AI Copilot with full context injection |

---

## 🧪 Data Strategy

- **Isolation**: GroupKFold cross-validation by Patient ID (PID) — ensures no child appears in both train and test sets.
- **Imputation**: Median imputation for numeric features, mode for categorical.
- **Validation**: Strict column sanitization for XGBoost/CatBoost special character compatibility.
- **Feature Whitelisting**: V1 models use only clinically interpretable bedside features (~11–12 per target).
- **Temporal Windowing**: V2 models use sliding windows of 5 sequential visits per child.

---

## 🖼️ Report Images

Pre-generated, report-ready images are available in two themes:

| Folder | Theme | Contents |
|---|---|---|
| `images/` | Dark (dashboard-matching) | Bar charts, scatter plots, trajectories |
| `images_light/` | Light (report/paper-ready) | Tabular comparisons, scatter plots, trajectories |

Each image has a companion `.txt` file with a brief clinical explanation.

---

## 🛠️ Key Scripts Reference

| Script | Purpose |
|---|---|
| `registry_precompute_v1.py` | Generate all V1 registry artifacts (predictions, SHAP, metadata) |
| `build_v1_pipeline.py` | Define strict clinical feature subsets per target |
| `build_multi_target_pipeline.py` | Multi-target pipeline builder with shared infrastructure |
| `temporal_lab/train_sequence_models.py` | Train LSTM/TCN on windowed sequences |
| `temporal_lab/extract_embeddings.py` | Extract penultimate-layer embeddings from trained models |
| `temporal_lab/evaluate_all_paradigms.py` | Full Baseline → V1 → Sequence → Hybrid benchmark |
| `stress_tests.py` | Automated leakage detection and feature validation |
| `run_feature_validation_fast.py` | Fast feature sanity checks |

---

## 🔧 Running the Platform

### Backend
```bash
cd backend
python main.py
# Runs on http://localhost:8000
```

### Frontend
```bash
cd ui
npm install
npm run dev
# Runs on http://localhost:5173
```

### Prerequisites
- Python 3.10+ with: `fastapi`, `uvicorn`, `pandas`, `numpy`, `scikit-learn`, `xgboost`, `catboost`, `shap`, `torch`
- Node.js 18+ with: `react`, `vite`, `recharts`, `axios`

---

*Last updated: April 26, 2026*
