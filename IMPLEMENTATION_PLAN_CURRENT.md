# Implementation Plan: Diagnostic Studio Expansion

## 🎯 Current Goal
Transform the **Diagnostic Studio** from a basic metric viewer into a comprehensive **Clinical Intelligence Suite**. This involves adding advanced Explainable AI (XAI) visualizations (SHAP Beeswarm/Force Plots) and a robust Graph Explorer for both Regression and Classification models.

---

## ✅ Completed So Far

### 1. Backend API Expansion
- Added `get_shap_detail` to `RegistryManager` to serve per-row SHAP values aligned with actual feature values.
- Added `get_classification_curves` to serve server-computed ROC, Precision-Recall, and Calibration data.
- Created new endpoints:
    - `GET /api/models/registry/shap_detail`
    - `GET /api/models/registry/classification_curves`

### 2. Frontend Architecture (Phase 1)
- **Modular Refactor**: Rewrote `DiagnosticStudio.jsx` into a modular structure.
- **Shared Components**: Created `ModelSelector.jsx`, `TargetSelector.jsx`, and `VersionSelector.jsx` in `ui/src/components/`.
- **Caching Engine**: Implemented `useRef` based caching to ensure each configuration is fetched only once per session.
- **Lazy Loading**: Set up logic to fetch heavy SHAP data only when the SHAP tab is active.
- **UI Scaffold**: Implemented the dual-tab system (SHAP Suite | Graph Explorer) and sub-tab routing.

---

## 🚀 Upcoming Phases

### Phase 2: SHAP Suite Implementation
- **SHAP Importance**: Polished bar chart.
- **SHAP Beeswarm**: Custom scatter plot with color gradients (Blue=Low, Red=High feature value).
- **SHAP Dependence**: Interactive feature selector to visualize impact across value ranges.
- **SHAP Force Plot**: Local explanation for individual samples.

### Phase 3: Graph Explorer (Regression)
- **Prediction vs Actual**: With diagonal parity line.
- **Residual Plot**: To detect heteroscedasticity/bias.
- **Error Distribution**: Histogram of residuals.
- **Time-Series Trajectory**: Visualizing PID-specific growth curves.

### Phase 4: Graph Explorer (Classification)
- **ROC & PR Curves**: Using server-computed points.
- **Interactive Confusion Matrix**: Tied to a **Threshold Slider (0-1)**.
- **Calibration Curve**: To assess probability reliability.

### Phase 5: Design & Polish
- Applying **Glassmorphism** styles to all new charts.
- Adding smooth transitions between sub-tabs.
- Ensuring strict data validation (never rendering on empty/undefined data).

---

## 🛠️ Tech Stack
- **Charts**: Recharts (Customized for SHAP Beeswarm).
- **Backend**: Python (Pandas + Sklearn for curve computation).
- **Frontend**: React (Vite).
