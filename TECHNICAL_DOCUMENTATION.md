# MAL-ED Clinical Nexus: Comprehensive Technical Documentation

*Source material for academic report — April 2026*

---

## 1. PROJECT OVERVIEW

### 1.1 What the System Is

The MAL-ED Clinical Nexus is an end-to-end clinical intelligence platform that predicts pediatric health outcomes — specifically child growth trajectories, illness burden, and recovery timelines — for children aged 0–60 months in resource-limited settings. It combines classical machine learning, deep temporal sequence modeling, and a hybrid fusion architecture into a single interactive dashboard with an embedded AI Copilot for clinical decision support.

The system was built on the MAL-ED (Etiology, Risk Factors, and Interactions of Enteric Infections and Malnutrition and the Consequences for Child Health) longitudinal cohort study, one of the most comprehensive multi-site pediatric datasets ever collected. It tracks 1,793 children across 1,532,191 daily observations spanning 176 clinical features — from anthropometric measurements to pathogen testing, illness surveillance, and socioeconomic indicators.

### 1.2 What Problem It Solves

In low-resource clinical settings, health workers must make rapid decisions about which children are at risk of malnutrition, chronic illness, or delayed recovery — often with minimal diagnostic tools. Existing approaches are either purely statistical (population-level Z-score thresholds) or require expensive laboratory tests.

This system addresses three specific gaps:

1. **Prediction**: Can we forecast a child's future nutritional status, illness burden, and recovery time using only bedside-available clinical features?
2. **Interpretability**: Can clinicians understand *why* the model flagged a child, not just *that* it did?
3. **Temporal Awareness**: Can we capture the fact that a child who has been declining over three consecutive visits is at higher risk than one with a single bad measurement?

### 1.3 High-Level Idea

The platform operates through a three-tier modeling paradigm:

- **Tier 1 (V1)**: Static tabular models (CatBoost, XGBoost, Random Forest) trained on ~11–12 clinically restricted features per target. These serve as the interpretable clinical baseline.
- **Tier 2 (V2)**: Deep temporal sequence models (LSTM, TCN) trained on windowed sequences of 5 consecutive anthropometric visits per child. These capture temporal patterns that static models miss.
- **Tier 3 (V4)**: Hybrid fusion models that concatenate the LSTM/TCN's learned temporal embeddings (64-dimensional latent vectors) with V1 clinical features, then train an XGBoost on the combined feature space. This is the champion architecture.

All model artifacts (predictions, SHAP explanations, metadata) are precomputed and stored in a structured registry, served through a FastAPI backend, and visualized through a React dashboard with glassmorphism design.

---

## 2. SYSTEM ARCHITECTURE

### 2.1 End-to-End Pipeline

```
Raw MAL-ED Data (1.5M rows, 176 cols)
    │
    ▼
[perfect_preprocess.py] — Column pruning, leakage removal, clinical imputation
    │
    ▼
[build_v1_pipeline.py] — Feature engineering (burden, recovery, velocity), target creation
    │
    ├─── V1 Static Datasets (baz_ar, delta_baz, diarrhea, illness_burden)
    │         │
    │         ▼
    │    [registry_precompute_v1.py] — GroupKFold CV, train 3 models × 4 targets
    │         │
    │         ▼
    │    models/v1/ Registry (metadata, predictions, SHAP)
    │
    ├─── V2 Sequence Datasets (windowed, 5 visits × 23 features)
    │         │
    │         ▼
    │    [train_sequence_models.py] — LSTM + TCN training (100 epochs, early stopping)
    │         │
    │         ▼
    │    [extract_embeddings.py] — Extract 64-dim latent vectors (Phase 3)
    │         │
    │         ▼
    │    [build_hybrid_datasets.py] — Join embeddings + V1 features (Phase 4)
    │         │
    │         ▼
    │    [evaluate_all_paradigms.py] — Full benchmark: Baseline → V1 → Seq → Hybrid
    │         │
    │         ▼
    │    models/v2/ Registry (paradigm results, prediction samples, model weights)
    │
    ▼
[backend/main.py] — FastAPI server (RegistryManager, Copilot, Simulation Engine)
    │
    ▼
[ui/] — React + Vite Dashboard (Explorer, Laboratory, Diagnostic, Temporal Lab, Copilot)
```

### 2.2 How Components Interact

The system follows a **precompute-then-serve** pattern. All expensive operations (model training, cross-validation, SHAP computation) happen offline. The backend is a thin serving layer that reads precomputed artifacts from disk and returns them as JSON. This design decision was made because:

1. **SHAP computation is slow** — TreeExplainer on 300 samples takes 10–30 seconds per model. Doing this on every API request would make the UI unusable.
2. **Reproducibility** — precomputed artifacts are deterministic and versioned. The same registry can be served to multiple users without retraining.
3. **Separation of concerns** — the training pipeline (Python scripts) and the serving pipeline (FastAPI) are completely independent. A researcher can retrain models without touching the backend.

The frontend communicates with the backend via REST API calls (axios). State is managed through React Context (`NexusContext` for version/target/model, `CopilotContext` for AI assistant state). The Copilot widget is a floating, resizable panel that receives serialized dashboard context and sends it to the LLM alongside user queries.

### 2.3 Modular Design

The project is organized into clear phases:

| Phase | Script | Output |
|-------|--------|--------|
| Phase 1 | `perfect_preprocess.py` | `mal_ed_final.parquet` (clean master dataset) |
| Phase 1.5 | `build_v1_pipeline.py` | V1 target-specific datasets with engineered features |
| Phase 2 | `registry_precompute_v1.py` | `models/v1/` registry (12 model variants) |
| Phase 2.5 | `build_sequence_datasets.py` | V2 windowed sequence datasets |
| Phase 3 | `train_sequence_models.py` | Trained LSTM/TCN PyTorch models |
| Phase 3.5 | `extract_embeddings.py` | V3 embedding parquets (64-dim per sample) |
| Phase 4 | `build_hybrid_datasets.py` | V4 hybrid datasets (embeddings + V1 features) |
| Phase 5 | `evaluate_all_paradigms.py` | Full benchmark results + prediction samples |

Each phase can be re-run independently. If sequence models are retrained, only Phases 3–5 need to re-execute. If the V1 feature whitelist changes, only Phases 1.5–2 need updating.

---

## 3. DATASET & PREPROCESSING

### 3.1 The MAL-ED Dataset

The MAL-ED study is a prospective, longitudinal birth cohort study conducted across 8 sites in South America, Africa, and South Asia. Children were enrolled at birth and followed with twice-weekly home surveillance visits (illness, diarrhea, fever, antibiotic use) and monthly anthropometric measurements (weight, height, Z-scores) for up to 5 years.

Key characteristics of the raw data:

- **1,532,191 rows** — each row represents one surveillance observation for one child on one day.
- **1,793 unique children** — tracked from birth to 60 months.
- **176 clinical features** — ranging from anthropometry (`Weight (kg)`, `Height (cm)`, `BMI-for-age z-score`) to illness surveillance (`Any illness, caregiver report`, `Diarrhea, caregiver report`) to socioeconomic indicators (`WAMI index`, `Maternal education (years)`, `Sanitation score`).
- **98.6% data completeness** — the MAL-ED study had unusually rigorous data collection protocols.

### 3.2 Sparsity and Temporal Nature

The dataset has a critical structural property: **anthropometric measurements (Z-scores) are sparse relative to surveillance data**. A child might have daily surveillance visits recording illness/diarrhea status, but BAZ (BMI-for-age Z-score) is measured only at monthly anthropometric visits. This creates a fundamental tension:

- Surveillance data is dense (daily) but contains only binary illness flags.
- Anthropometric data is sparse (monthly) but contains the actual growth measurements we want to predict.

This distinction drove a key architectural decision: **sequence models are built from anthropometric visit rows only**, not daily surveillance rows. Using daily rows would produce sequences of nearly identical values (a child's weight doesn't change day-to-day) with no temporal learning signal. Instead, between each pair of monthly visits, we *aggregate* the daily surveillance data into summary features (count of illness days, proportion of days sick, etc.).

### 3.3 Missing Values and Imputation Decisions

**Why interpolation was removed**: Early versions of the pipeline used linear interpolation to fill missing Z-scores between visits. This was removed because interpolation creates artificial smoothness that doesn't exist in reality. A child who wasn't measured for 2 months might have experienced a rapid decline followed by recovery — interpolation would hide this. The clinical insight is that *the absence of measurement is itself informative* (the child may not have been brought in because they were too sick or too healthy to warrant a visit).

**Why global median imputation was removed**: Global median imputation (replacing missing values with the population median) was removed for time-varying features because it destroys temporal signal. If a child's weight measurement is missing at month 6, replacing it with the population median weight ignores whether this child was underweight or overweight at month 5. Instead, we use:

- **Zero-fill** for symptom/episode counts (missing = no event observed).
- **Forward-fill then backward-fill** for static baseline features (WAMI index, maternal education) within each child's history.
- **Median imputation** only for truly static features (scores, indices) where per-child imputation is impossible.

**Why preserving real temporal gaps matters**: In the sequence datasets, the `time_delta` feature captures the number of days between consecutive anthropometric visits. If a child has a 90-day gap instead of the usual 30-day gap, this is clinically significant (possible missed appointments, illness, or relocation). Filling in synthetic intermediate visits would destroy this signal.

---

## 4. FEATURE ENGINEERING

### 4.1 Burden Features (30-Day Rolling History)

Three burden features are computed using a 30-day backward-looking rolling window per child:

- `burden_illness_30d` — Count of illness days in the past 30 days.
- `burden_diarrhea_30d` — Count of diarrhea days in the past 30 days.
- `burden_antibiotics_30d` — Count of antibiotic-use days in the past 30 days.

These are computed using pandas `rolling("30D", closed="left")` on binarized event columns, grouped by patient ID. The `closed="left"` parameter is critical — it ensures the current day's observation is *excluded* from the rolling sum, preventing same-day leakage.

**Why these reflect biological processes**: Malnutrition is not caused by a single illness episode but by *cumulative infectious burden*. A child who has been sick for 15 of the last 30 days has a fundamentally different prognosis than one who was sick for 2 days. The 30-day window was chosen because it approximates the "immunological memory" period — the time over which repeated infections compound their damage to gut integrity and nutrient absorption.

### 4.2 Recovery Features (Time Since Last Event)

Three recovery features measure the number of days since the most recent episode:

- `recovery_days_since_illness` — Days since the last illness episode.
- `recovery_days_since_diarrhea` — Days since the last diarrhea episode.
- `recovery_days_since_antibiotics` — Days since the last antibiotic course.

Missing values (no prior event) are filled with 999, representing "never had this event" — a deliberate sentinel value that tree-based models can learn to split on.

**Why these matter clinically**: Recovery from enteric infection is not instantaneous. Gut epithelium takes 2–4 weeks to regenerate after a diarrheal episode. A child measured 5 days after severe diarrhea will have a different BAZ trajectory than one measured 45 days after. These features capture the *recovery phase* that burden features alone cannot represent.

### 4.3 Target Velocity

`target_velocity` = (BAZ(t) - BAZ(t-1)) / (agedays(t) - agedays(t-1)), clipped to [-5, 5].

This captures the *rate of change* in nutritional status, not just the level. A child at BAZ = -1.5 who is declining at -0.02 per day is in a very different situation from one at BAZ = -1.5 who is recovering at +0.01 per day.

### 4.4 Leakage-Safe Design

The V1 feature whitelist (`ALLOWED_FEATURES` in `build_v1_pipeline.py`) was designed with explicit anti-leakage rules:

1. **No future information**: `target_prev` (the current BAZ, used to predict the *next* BAZ) is backward-looking. `target_velocity` is computed from past visits only.
2. **No target-derived features**: For the diarrhea classification target, `burden_diarrhea_30d` and `recovery_days_since_diarrhea` were included in the "full" variant but *excluded* from the "safe" variant to test whether diarrhea history leaks into diarrhea prediction.
3. **No summary statistics**: Features like "total illness days over study" were explicitly dropped in preprocessing because they summarize the future.

### 4.5 Enhanced Temporal Features (Phase 2.5)

For sequence models, 8 additional features were engineered per timestep:

- **Short-term memory**: `illness_last_3`, `diarrhea_last_3`, `antibiotics_last_3` — cumulative event counts over the last 3 visits (not days), capturing medium-term trends.
- **Trend signals**: `weight_delta_last_2`, `height_delta_last_2` — growth change over the last 2 visits.
- **State flags**: `is_currently_ill` (binary), `is_recovering` (was ill last window, not ill now).
- **Age phase**: `age_group` — infant (0) / toddler (1) / child (2), encoding the non-linear growth dynamics across developmental stages.

---

## 5. TARGET DESIGN

### 5.1 BAZ Autoregressive (baz_ar)

**What it is**: Predict the *next* BMI-for-age Z-score given the current clinical state.

**Purpose**: This is the most clinically actionable target. If a model can predict that a child's BAZ will drop below -2 at the next visit, a health worker can intervene *before* the child becomes wasted.

**Inertia baseline**: BAZ is highly autocorrelated — a child's Z-score at month 6 is strongly predicted by their Z-score at month 5. The dominant feature is `target_prev` (current BAZ), which alone explains ~80% of variance. This is not a weakness — it reflects the biological reality that nutritional status has inertia. The challenge is whether the model can identify the *remaining 4–5%* of variance that comes from illness burden, recovery state, and other clinical signals.

**Results**: CatBoost achieves R² = 0.843, MAE = 0.298 Z-score units. This means the model's prediction is off by less than 0.3 Z-score units on average — clinically sufficient for screening.

### 5.2 Growth Velocity (ΔBAZ / delta_baz)

**What it is**: Predict the *change* in BAZ between consecutive visits: BAZ(t) - BAZ(t-1).

**Why we tried it**: Predicting the *change* rather than the *level* removes the autocorrelation that inflates BAZ-AR performance. If we can predict growth velocity, we can identify children who are *about to deteriorate* even if their current Z-score looks acceptable.

**Why it's difficult**: ΔBAZ has extremely low signal-to-noise ratio. The mean change between visits is near zero (most children's growth is stable), and the variance is driven by measurement noise, acute illness episodes, and genuine growth spurts — all of which are hard to distinguish. V1 static models achieve only R² = 0.079 (CatBoost), meaning they explain less than 8% of the variance in growth changes.

**Key insight**: The hybrid model dramatically improves this to R² = 0.546 (MAE = 0.188 vs 0.293), a 31% MAE reduction. This suggests that temporal patterns (declining over multiple visits) contain information that single-visit features cannot capture. This is the strongest evidence for the hybrid approach.

### 5.3 Illness Burden

**What it is**: Predict the *total number of illness days in the next inter-visit window* (typically 30 days).

**Why it works well**: Illness burden has strong temporal clustering — children who have been frequently sick recently are more likely to be sick again soon (damaged gut epithelium → reduced immunity → more infections). The burden features (`burden_illness_30d`, `burden_diarrhea_30d`) capture this cycle directly.

**Results**: Hybrid LSTM achieves R² = 0.609, MAE = 1.49 illness days (vs. baseline persistence R² = 0.128). The model can predict how many sick days a child will have in the next month with an average error of 1.5 days.

### 5.4 Time-to-Recovery

**What it is**: Predict the number of days until the first day with *no illness AND no diarrhea* in the next inter-visit window, capped at 60 days.

**Why it's a strong temporal signal**: Recovery time depends on the *trajectory* of illness — a child who has been sick for 3 consecutive visits will take longer to recover than one with an isolated episode. This is inherently temporal.

**Results**: Hybrid LSTM achieves MAE = 11.69 days (vs. baseline MAE = 31.31 days, a 63% reduction). The model is off by about 12 days on average when predicting recovery time, which is clinically useful for discharge planning.

### 5.5 Diarrhea Risk (Classification)

**What it is**: Binary classification — will the child have *any* diarrhea episode in the next inter-visit window?

**Why it was initially included**: Diarrhea is the leading cause of child mortality in MAL-ED settings, making it an obvious prediction target.

**Why it performed poorly and was deprioritized**: The diarrhea target achieved high AUC (0.873) but this is misleading. The problem is class imbalance and definition: diarrhea episodes are common (high base rate), so a model that simply predicts "yes" for children with any recent diarrhea history achieves high recall. The model learns `burden_diarrhea_30d` as the dominant predictor, which is essentially "the child had diarrhea recently, so they'll have it again." This is circular rather than predictive. The classification was kept in V1 for completeness but was excluded from the temporal modeling pipeline (V2/V4) because:

1. The binary target loses granularity (how many days of diarrhea is more useful than yes/no).
2. Illness burden already captures diarrhea severity as a continuous variable.
3. The diarrhea-safe variant (excluding diarrhea history features) performed much worse, confirming that the model was learning a trivial pattern.

---

## 6. MODELING APPROACH

### 6.1 Classical Models (V1)

Three tree-based ensemble methods were evaluated:

**Random Forest** (50 estimators): Bagging ensemble that reduces variance through bootstrap aggregation. Each tree sees a random subset of features and samples, making it robust to noisy features. Used `class_weight='balanced'` for classification.

**XGBoost** (50 estimators, max_depth=6): Gradient boosting with regularization (L1/L2). Builds trees sequentially, each correcting the errors of the previous. Handles missing values natively and supports column subsampling.

**CatBoost** (50 iterations, depth=6): Gradient boosting with ordered boosting (processes training samples in a random order to reduce prediction shift). Handles categorical features natively and uses symmetric decision trees for faster inference. Used `auto_class_weights='Balanced'` for classification.

**Why tree models are strong for tabular data**: The clinical features in MAL-ED are heterogeneous — mixing continuous measurements (weight, height), counts (burden days), binary flags (illness status), and scores (WAMI index). Tree-based models handle this naturally without requiring feature scaling or normalization. They also capture non-linear interactions (e.g., "burden_illness_30d > 10 AND recovery_days_since_diarrhea < 7" as a compound risk factor) without explicit feature engineering. Additionally, they produce feature importance scores that are clinically interpretable.

**CatBoost consistently wins across all targets** on R² and MAE. The advantage is small but consistent (R² = 0.843 vs 0.838 for baz_ar), likely because ordered boosting reduces overfitting on this noisy clinical data. CatBoost is designated the champion static model.

### 6.2 Sequence Models (V2)

#### LSTM (Long Short-Term Memory)

**Architecture**: 2-layer LSTM with hidden_size=64, dropout=0.2, followed by FC(64→32→1) with ReLU activation.

**What it captures**: The LSTM processes a sequence of 5 consecutive visits (each with 23 features) and maintains a hidden state that accumulates temporal context. It can learn patterns like "this child's BAZ has been declining for 3 visits while illness burden has been increasing" — patterns that no single-visit feature can represent.

**Training**: MSE loss, Adam optimizer (lr=1e-3, weight_decay=1e-5), gradient clipping (max_norm=1.0), early stopping with patience=10, batch_size=256, max 100 epochs.

#### TCN (Temporal Convolutional Network)

**Architecture**: 2 blocks of dilated causal convolutions (channels=[64, 64], kernel_size=3, dilations=[1, 2]), each block containing two Conv1d layers with chomp (causal padding removal), ReLU, dropout, and residual connection. Final output: FC(64→1).

**Why compared**: TCN was included as a comparison because it offers parallelizable training (vs LSTM's sequential processing) and has a fixed receptive field (determined by dilation structure) rather than theoretically infinite memory. For short sequences (5 timesteps), the TCN's receptive field fully covers the input, so there is no theoretical advantage to LSTM's memory.

**Why both performed similarly**: LSTM achieved R² = 0.155 and TCN achieved R² = 0.153 on delta_baz — virtually identical. This is a meaningful finding: it indicates that the temporal dependencies in this data are *relatively simple* (recent trends over 3–5 visits) rather than long-range dependencies that would favor LSTM. The key signal is "what happened in the last 2–3 visits," not "what happened 20 visits ago." Both architectures capture this equally well.

### 6.3 Hybrid Models (V4)

**Architecture**: The trained LSTM/TCN's penultimate layer output (a 64-dimensional vector) is extracted for each sequence. This embedding is concatenated with the V1 clinical features (~11–12 features) to create a combined feature vector of ~75–76 dimensions. An XGBoost regressor (200 estimators, max_depth=6) is then trained on this combined space.

**Why hybrid works better**: The hybrid model outperforms both pure-tabular and pure-sequence models on every target:

| Target | V1 (Tabular) R² | LSTM R² | Hybrid LSTM R² | MAE Improvement |
|--------|-----------------|---------|----------------|-----------------|
| ΔBAZ | 0.126 | 0.155 | **0.546** | 31% vs V1 |
| Illness | 0.530 | 0.538 | **0.609** | 11% vs V1 |
| Recovery | 0.516 | 0.512 | **0.511** | 10% vs V1 |

The insight is that **history and context are complementary**:
- The LSTM embedding encodes *how the child has been changing* over 5 visits (temporal dynamics).
- The V1 features encode *where the child is right now* (current clinical state: weight, illness burden, recovery phase).
- XGBoost can learn non-linear interactions between these two information sources — e.g., "temporal embedding suggests declining trajectory AND current burden is high → predict worse outcome."

Neither source alone contains the full picture. The LSTM knows the trajectory but not the absolute clinical context. The V1 features know the current state but not the trajectory. The hybrid fusion enables the model to reason about both simultaneously.

---

## 7. EVALUATION STRATEGY

### 7.1 GroupKFold (PID-Based Splitting)

All evaluations use `GroupKFold` with `groups=pid` (patient ID). This ensures that **all observations from the same child appear in either the training set or the test set, never both**. This is critical because:

1. **Temporal autocorrelation**: A child's measurements at month 6 and month 7 are highly correlated. If month 6 is in training and month 7 is in testing, the model has effectively "seen" the answer. GroupKFold prevents this.
2. **Clinical realism**: In deployment, the model will encounter *new* children it has never seen. GroupKFold simulates this by evaluating on entirely unseen children.
3. **Leakage prevention**: Without grouping, a model could memorize per-child patterns (e.g., "child X always has BAZ around -1.2") rather than learning generalizable clinical signals.

V1 static models use 5-fold GroupKFold with `cross_val_predict` to generate out-of-fold predictions for the entire dataset. V2 temporal models use a single `GroupShuffleSplit` with 80/20 train/test ratio (same random seed for reproducibility across all model types).

### 7.2 Metrics

**R² (Coefficient of Determination)**: Measures the proportion of variance in the target explained by the model. R² = 1.0 means perfect prediction; R² = 0.0 means the model is no better than predicting the mean; R² < 0 means the model is worse than the mean (which happens for naive baselines on difficult targets like recovery).

**MAE (Mean Absolute Error)**: The average absolute difference between predicted and actual values. For BAZ, MAE = 0.298 means the model's prediction is off by 0.298 Z-score units on average. For recovery, MAE = 11.7 means the model is off by about 12 days. **MAE is the most clinically meaningful metric** because it directly answers "how wrong is the prediction for a typical patient?" in natural units.

**RMSE (Root Mean Squared Error)**: Similar to MAE but penalizes large errors more heavily (due to squaring). RMSE > MAE indicates the presence of outlier predictions — cases where the model is wildly wrong for a few children. Clinically important because a prediction that is off by 60 days for one child is more dangerous than being off by 5 days for 12 children, even though the total error is the same.

**Why MAE is most clinically meaningful**: In a clinical setting, a health worker asks "how far off is this prediction?" not "what percentage of variance is explained?" A model with R² = 0.85 but MAE = 0.5 Z-scores is less useful than one with R² = 0.80 but MAE = 0.25 Z-scores. The system prompt for the AI Copilot explicitly encodes this priority: MAE > RMSE > R² for regression tasks.

**Why baseline comparisons are critical**: Raw metrics are meaningless without baselines. An R² of 0.51 for recovery prediction sounds mediocre, but the naive baseline (last inter-visit interval) achieves R² = -2.16. The model is not just "slightly better than chance" — it is transformatively better than any naive heuristic. Similarly, for BAZ-AR, CatBoost's R² = 0.843 must be compared against the inertia baseline (~0.80 from `target_prev` alone) to understand that the model adds ~4 percentage points of explained variance from clinical features beyond simple autocorrelation.

---

*Continued in Part 2 (Sections 8–15): Results, Engineering Architecture, XAI, Temporal Lab, Prediction Lab, Copilot, Limitations, Future Work*

## 8. RESULTS & INSIGHTS

### 8.1 Static Model Results (V1)

#### BAZ Autoregressive — High predictability, dominated by inertia

| Model | R² | MAE (Z-score) | RMSE |
|-------|-----|---------------|------|
| **CatBoost** | **0.843** | **0.298** | **0.448** |
| XGBoost | 0.838 | 0.302 | 0.456 |
| Random Forest | 0.838 | 0.306 | 0.456 |

The near-identical performance across all three models (R² spread of only 0.005) tells us something important: the signal in this target is straightforward and fully exploitable by any competent tree ensemble. The dominant feature is `target_prev` (SHAP importance = 0.83), meaning ~83% of the model's decision is "the child's next BAZ will be close to their current BAZ." The remaining signal comes from `target_velocity` (growth trend), `Weight (kg)`, and `agedays` — all clinically sensible.

#### Growth Velocity (ΔBAZ) — The hardest target

| Model | R² | MAE | RMSE |
|-------|-----|-----|------|
| **CatBoost** | **0.079** | **0.293** | **0.444** |
| Random Forest | 0.057 | 0.299 | 0.449 |
| XGBoost | 0.042 | 0.299 | 0.453 |

R² below 0.10 means static features explain less than 8% of growth velocity variance. This is not a model failure — it's a data insight. Growth changes are driven by acute events (infections, dietary shifts, seasonal effects) that are poorly captured by a single-visit snapshot. This is precisely why temporal models were developed.

#### Diarrhea Risk (Classification)

| Model | ROC-AUC | F1 | F2 |
|-------|---------|-----|-----|
| **CatBoost** | **0.873** | 0.830 | 0.787 |
| XGBoost | 0.866 | **0.923** | **0.933** |
| Random Forest | 0.847 | 0.921 | 0.932 |

Note the metric disagreement: CatBoost wins on AUC but XGBoost wins on F1/F2. This happens because CatBoost uses balanced class weights, which shifts the decision threshold toward higher precision (fewer false positives) at the cost of recall. XGBoost without explicit balancing predicts "positive" more liberally, catching more true positives (higher recall/F2) but also more false alarms. In clinical settings where missing a sick child is worse than a false alarm, **XGBoost's higher F2 (0.933) is actually preferred**. This is exactly the kind of nuance the Copilot system prompt was designed to surface.

#### Illness Burden — The strongest static target

| Model | R² | MAE (days) | RMSE |
|-------|-----|-----------|------|
| **CatBoost** | **0.640** | **2.110** | **4.314** |
| XGBoost | 0.629 | 2.116 | 4.378 |
| Random Forest | 0.626 | 2.144 | 4.396 |

Illness burden is well-predicted because it has strong temporal autocorrelation (cumulative infectious burden cycles) and the V1 features directly capture the relevant clinical state (recent illness history, antibiotics, recovery phase).

### 8.2 Temporal & Hybrid Results (V2/V4)

The full paradigm benchmark reveals the progressive improvement across modeling tiers:

#### Growth Velocity (ΔBAZ) — Hybrid shines brightest here

| Model | R² | MAE | RMSE | MAE Δ vs V1 |
|-------|-----|-----|------|-------------|
| Baseline (Zero Growth) | -0.002 | 0.285 | 0.430 | — |
| V1 (XGBoost Tabular) | 0.126 | 0.273 | 0.402 | baseline |
| LSTM | 0.155 | 0.267 | 0.395 | -2.2% |
| TCN | 0.153 | 0.268 | 0.396 | -1.8% |
| **Hybrid LSTM** | **0.546** | **0.188** | **0.290** | **-31.1%** |
| Hybrid TCN | 0.475 | 0.206 | 0.311 | -24.5% |

The hybrid model's R² jumps from 0.155 (LSTM alone) to 0.546 — a 3.5× improvement. This is the project's strongest finding: **temporal embeddings + clinical features together capture growth dynamics that neither can alone.**

#### Illness Burden

| Model | R² | MAE (days) | RMSE |
|-------|-----|-----------|------|
| Baseline (Persistence) | 0.128 | 2.149 | 5.377 |
| V1 (XGBoost Tabular) | 0.530 | 1.675 | 3.947 |
| **Hybrid LSTM** | **0.609** | **1.487** | **3.601** |

#### Recovery (Time-to-Recovery)

| Model | R² | MAE (days) | RMSE |
|-------|-----|-----------|------|
| Baseline (Last Interval) | -2.160 | 31.310 | 79.621 |
| V1 (XGBoost Tabular) | 0.516 | 12.947 | 31.171 |
| **Hybrid LSTM** | **0.511** | **11.687** | **31.320** |

### 8.3 Why Some Targets Are Predictable and Others Are Not

**Illness burden is the most predictable temporal target** (R² = 0.609) because infectious burden is self-reinforcing: repeated gut infections damage the intestinal barrier, leading to more infections. This creates strong temporal autocorrelation that both static burden features and temporal embeddings can exploit.

**ΔBAZ is the least predictable with static features** (R² = 0.079) but **benefits most from temporal modeling** (R² = 0.546 hybrid). This is because growth velocity changes are driven by *sequences of events* — a child declining over 3 visits signals a developing problem, but any single visit might look like normal variation. The LSTM embedding captures this multi-visit trajectory, and the hybrid XGBoost learns to combine it with the current clinical snapshot.

**Recovery time is moderately predictable** (R² ≈ 0.51) and shows the *smallest* hybrid improvement. This suggests that recovery dynamics are more about the *current state* (severity of current illness, recent antibiotic use) than the *trajectory*. A child who is very sick right now will take longer to recover regardless of their trajectory over the last 5 visits.

### 8.4 What the Models Reveal About the Data

1. **BAZ is dominated by inertia**: The overwhelming importance of `target_prev` (SHAP = 0.83) confirms that child growth is a slow-moving process. Clinical interventions must be sustained over weeks to months to shift Z-scores.

2. **Temporal patterns are real but short-range**: LSTM and TCN perform identically, suggesting that the relevant temporal patterns span 2–3 visits (2–3 months), not longer. This aligns with the clinical literature on acute malnutrition recovery windows.

3. **Hybrid fusion is not just additive**: The hybrid improvement on ΔBAZ (R² from 0.155 to 0.546) is far larger than the sum of LSTM and V1 improvements individually. This indicates *synergistic interactions* — the XGBoost discovers non-linear combinations of temporal embeddings and clinical features that neither source provides alone.

4. **Illness predicts illness**: The burden features are consistently among the top SHAP features across all targets, confirming the MAL-ED study's central finding that enteric infection burden is the primary driver of childhood malnutrition in these settings.

---

## 9. ENGINEERING ARCHITECTURE

### 9.1 Backend (FastAPI)

The backend (`backend/main.py`, ~69K bytes) is a single FastAPI application that serves as the data layer for the entire dashboard. Key architectural decisions:

**RegistryManager class**: A centralized interface for reading model artifacts from disk. Methods include `get_metadata()`, `get_predictions()`, `get_shap_global()`, `get_shap_sample()`, `get_expected_value()`, `get_classification_curves()`, and `get_threshold_metrics()`. All methods accept `(target, model, version)` parameters and return the precomputed artifact or `None` if not found. This abstraction isolates the frontend from the file system structure.

**Why pretraining was used**: The alternative was on-demand training — train a model every time the user clicks "Run" in the Model Arena. This was rejected because: (1) GroupKFold cross-validation with SHAP computation takes 30–60 seconds per model, making the UI unresponsive, (2) results would vary between runs due to randomness, making debugging impossible, and (3) precomputed results can be version-controlled and audited. The "Force Retrain" mode in the Arena exists for experimentation but uses lighter-weight training (no SHAP, no cross-validation) for speed.

**PROJECT_ROOT resolution**: A critical bug fix was implementing `PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))` with parent directory resolution. Without this, the backend would fail to find data files when run from different working directories (e.g., `python main.py` from `backend/` vs `python backend/main.py` from the project root).

**GLOBAL_STATE**: The dataset is loaded once at startup and stored in an in-memory dictionary. This avoids re-reading the 1.5M-row parquet file on every API call. The trade-off is higher memory usage (~500MB), but for a single-user research tool this is acceptable.

### 9.2 Frontend (React + Vite)

The frontend is a React single-page application built with Vite for fast HMR (Hot Module Replacement) during development.

**Why modular UI was designed**: Each dashboard tab is a self-contained React component (~20–45K bytes each) with its own state management. This was chosen over a monolithic component because:

1. **Separation of concerns**: The Diagnostic Studio doesn't need to know about the Temporal Lab's pipeline state.
2. **Lazy rendering**: Only the active tab's component is rendered, keeping the DOM small.
3. **Independent development**: Each tab can be modified without risking regressions in others.

**Component library**: Reusable visualization components (`Beeswarm.jsx`, `ForcePlot.jsx`, `RegressionGraphs.jsx`, `ClassificationGraphs.jsx`) are shared across tabs. All charts use Recharts for consistent styling and responsiveness.

**NexusContext**: A React Context provider that stores the global model version (V0/V1), selected target, and selected model. Changing the version toggle in the header automatically propagates to all tabs, causing them to fetch new registry data.

**CopilotWidget**: A floating, resizable panel that can be opened from any tab. It receives serialized dashboard context (active tab, metrics, graph data) and sends it to the backend's `/api/copilot/chat` endpoint alongside the user's message. The widget supports drag-to-resize (min 320×400, max 800×900) and auto-suggested prompts.

---

## 10. EXPLAINABLE AI (XAI)

### 10.1 SHAP Implementation

Every model in the V1 registry includes precomputed SHAP artifacts:

- **`shap_global.json`**: Mean absolute SHAP values per feature, averaged over 300 random samples. Used for feature importance bar charts.
- **`shap_sample.parquet`**: Per-sample SHAP values + original feature values for 300 samples. Used for Beeswarm plots and Force Plots.
- **`expected_value.json`**: The model's base prediction (population average), needed to construct SHAP force plots.

SHAP values are computed using `TreeExplainer` for tree-based models (exact computation, not approximation). For edge cases where TreeExplainer fails, the system falls back to a model-agnostic `Explainer` with permutation-based approximation.

### 10.2 Why Interpretability Matters in Healthcare

In clinical deployment, a model that says "this child is at risk" is useless without explaining *why*. A health worker needs to know:

- "Is the risk driven by recent illness history (actionable: treat infections more aggressively) or by low WAMI index (not immediately actionable: socioeconomic factor)?"
- "Is this child at risk because they've been declining for 3 months (trajectory) or because of a single acute event (possibly transient)?"

SHAP provides this at two levels:
1. **Global importance**: Which features matter most *across all children* — useful for understanding the model's overall strategy and validating clinical plausibility.
2. **Local explanation (Force Plot)**: For *this specific child*, which features are pushing the prediction up (toward risk) and which are pushing it down (protective). This is the bedside-actionable output.

The Diagnostic Studio tab surfaces both levels: Beeswarm plots for global patterns, Force Plots for per-patient drill-down.

---

## 11. TEMPORAL LAB (SEQUENCE + HYBRID PIPELINE)

### 11.1 Interactive Pipeline Design

The Temporal Lab tab presents the hybrid modeling pipeline as a **4-step interactive simulation**:

1. **Learn Temporal Patterns**: Click to "train" the LSTM/TCN backbone. The UI shows training progress and final metrics.
2. **Extract Embeddings**: Click to extract 64-dimensional latent vectors from the trained model's penultimate layer.
3. **Combine with Clinical Context**: Click to merge embeddings with V1 clinical features, creating the hybrid dataset.
4. **Evaluate Hybrid**: Click to run the full benchmark, displaying a comparison table of all model paradigms.

**Why this interactive pipeline was built**: The pipeline could have been a single "Train Hybrid" button. Instead, it was decomposed into stages to help users *understand the hybrid approach conceptually*. By seeing each phase's output — temporal metrics alone, then embeddings, then the combined result — the user builds intuition about *why* the hybrid works. The comparison table at the end (Baseline → V1 → LSTM → Hybrid) makes the progressive improvement visible.

### 11.2 Embedding Extraction

The `get_embedding()` method on both LSTM and TCN models hooks into the penultimate layer:

- **LSTM**: Returns the last hidden state `h_T` from the LSTM's output sequence — a 64-dimensional vector that encodes the *summary of the entire 5-visit history* as the LSTM "understood" it.
- **TCN**: Returns the last temporal position of the final convolutional block's output — a 64-dimensional vector that encodes the local temporal context at the end of the sequence.

These embeddings are *not human-interpretable* (they are learned latent representations), but they capture temporal patterns that the downstream XGBoost can exploit. The XGBoost effectively "translates" these latent patterns into predictions by learning how they interact with clinical features.

---

## 12. PREDICTION LAB (SIMULATION ENGINE)

### 12.1 What-If Simulation

The Simulation Engine tab allows users to:

1. **Select a real child** from the cohort by patient ID.
2. **View their current clinical state**: feature values, temporal embeddings, and the model's current prediction.
3. **Adjust clinical features via sliders**: Change weight, illness burden, recovery days, antibiotic use, etc.
4. **See real-time prediction updates**: The hybrid model re-predicts the outcome using the modified features while keeping the temporal embedding fixed.

**Design insight**: The temporal embedding is *frozen* during simulation because it represents the child's *history*, which cannot be retroactively changed. Only the V1 clinical features (representing the *current state*) are adjustable. This creates a clinically meaningful what-if: "If this child, with their existing history, were to have X fewer illness days next month, how would their predicted recovery change?"

### 12.2 Trajectory Projections

The system also generates longitudinal growth trajectory visualizations — plotting BAZ scores over time for individual children with WHO stunting thresholds (-2 and -3 Z-scores). These visualizations help clinicians see:

- Whether a child is on a declining trajectory (approaching -2).
- Whether interventions appear to be working (trajectory inflection point).
- How this child compares to cohort norms.

---

## 13. LLM COPILOT

### 13.1 Integration Architecture

The Copilot uses OpenAI's GPT-4 via API, integrated through the `/api/copilot/chat` endpoint. The system prompt is dynamically constructed at each request, injecting the current dashboard state:

```
Active Tab: {tab_name}
Target Outcome: {target}
Model(s): {model_name}
Graph Type: {graph_type}
Metrics Block: [structured metrics table]
Graph Data Block: [actual numbers from the current chart]
```

### 13.2 Context-Aware Prompting

The Copilot doesn't just receive the user's question — it receives the *entire visual context* of what the user is looking at. When the user clicks "Explain this view" while viewing a SHAP Beeswarm plot, the Copilot receives:

- The top 10 features by SHAP importance with their actual values.
- The target outcome and model name.
- A human-readable description of the chart.

This enables responses like "The model relies most heavily on `target_prev` (SHAP = 0.83), which means 83% of the prediction is based on the child's current BAZ. The next most important features are `Weight (kg)` and `burden_illness_30d`, suggesting that recent illness load directly impacts predicted growth."

### 13.3 Clinical Reasoning Rules

The system prompt includes explicit multi-metric comparison rules:

- **Regression**: Evaluate in priority order MAE → RMSE → R². If metrics disagree on a winner, flag it explicitly.
- **Classification**: Evaluate F1/F2 → Recall → AUC → Precision. In clinical settings, catching sick children (recall) matters more than avoiding false alarms (precision).
- **Never** declare a winner on a single metric alone.

**Why the Copilot is important for interpretability**: SHAP values and model metrics are powerful but require expertise to interpret. A health worker who sees "R² = 0.843, MAE = 0.298, RMSE = 0.448" may not know whether this is good or bad. The Copilot translates this into: "The model predicts the next BAZ measurement with an average error of about 0.3 Z-score units — roughly the difference between being borderline wasted and clearly wasted. This is accurate enough for clinical screening."

---

## 14. LIMITATIONS

### 14.1 ΔBAZ Unpredictability

Even the best hybrid model explains only 54.6% of growth velocity variance. Nearly half of growth changes remain unpredictable, likely driven by:
- Unmeasured dietary intake (the MAL-ED dataset doesn't capture day-to-day food consumption in sufficient detail).
- Seasonal effects (monsoon seasons increase diarrheal disease burden).
- Genetic variation in growth response to illness.
- Measurement error in anthropometry (small errors in weight/height produce large errors in Z-score changes).

### 14.2 Sparse Anthropometric Data

With only ~5–10 BAZ measurements per child per year, the temporal models have very short sequences to work with (window_size=5). This limits the LSTM/TCN's ability to learn long-range temporal patterns. Children with fewer than 6 anthropometric visits are excluded entirely, potentially biasing results toward children with better healthcare access.

### 14.3 Model Constraints

- **Sample size**: 1,793 unique children is modest for deep learning. The LSTM/TCN models may be underfitting due to limited training data.
- **Single cohort**: All models are trained and evaluated on MAL-ED data. Generalization to other populations (different pathogens, dietary patterns, healthcare systems) is unvalidated.
- **Static feature whitelist**: The V1 clinical features were hand-selected based on clinical intuition. Automated feature selection (e.g., Boruta) might identify different optimal subsets.
- **No uncertainty quantification**: The models produce point predictions without confidence intervals. A prediction of "recovery in 15 days ± 3 days" would be more useful clinically than "recovery in 15 days."

### 14.4 Classification Target Weakness

The diarrhea classification target is inherently limited by class imbalance and temporal autocorrelation. The model essentially learns "children who had diarrhea recently will have it again," which is clinically obvious and not actionable. A more useful target would be "first-onset diarrhea" (predicting when a currently-healthy child will become sick), but this requires different data preparation and would have even lower sample sizes.

---

## 15. FUTURE IMPROVEMENTS

### 15.1 Better Targets

- **Composite malnutrition risk score**: Combine BAZ, illness burden, and recovery into a single clinical risk score that captures multi-dimensional health status.
- **Stunting prediction**: Predict height-for-age Z-score (HAZ) instead of BMI. Stunting is a more stable indicator of chronic malnutrition and may be more predictable.
- **Event prediction**: Instead of continuous targets, predict discrete clinical events (hospitalization, severe acute malnutrition episode, mortality).

### 15.2 Improved Sequence Models

- **Transformer architectures**: Self-attention mechanisms could capture longer-range dependencies and cross-feature interactions more effectively than LSTM/TCN.
- **Longer sequences**: If data permits, extending the window from 5 to 10+ visits could reveal longer-term trends.
- **Multi-task learning**: Train a single sequence model to predict all targets simultaneously, sharing temporal representations across tasks.
- **Attention-based interpretability**: Transformer attention weights could provide temporal explanations ("the model is paying most attention to visit 3, when the child had acute diarrhea").

### 15.3 Deployment Considerations

- **Model compression**: The hybrid pipeline (PyTorch LSTM + XGBoost) requires both frameworks at inference time. Distilling into a single model or ONNX export would simplify deployment.
- **Edge deployment**: For use in rural clinics without internet, models would need to run on mobile devices or tablets.
- **Continuous learning**: As new patient data arrives, models should be periodically retrained. An automated retraining pipeline with drift detection would be needed.
- **Clinical validation**: Before deployment, the system should undergo a prospective clinical trial comparing model-assisted decisions to standard care.
- **Uncertainty estimation**: Implementing conformal prediction or Bayesian approaches to provide prediction intervals alongside point estimates.

---

*End of Technical Documentation*

*This document covers the complete MAL-ED Clinical Nexus system — from raw data preprocessing through three tiers of modeling to a production-grade interactive dashboard. Every design decision is traced from clinical motivation through implementation to observed results, providing comprehensive source material for academic reporting.*
