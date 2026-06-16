# MAL-ED Clinical Nexus: Results & Insights Documentation

*Comprehensive results analysis for academic report — April 2026*

---

## 1. OVERVIEW OF EXPERIMENTAL SETUP

### 1.1 Prediction Targets

The system predicts three temporal regression targets and one classification target:

- **ΔBAZ (Growth Velocity)**: The change in BMI-for-age Z-score between consecutive anthropometric visits. Captures whether a child is improving or deteriorating nutritionally.
- **Illness Burden**: The total number of illness days in the next inter-visit window (~30 days). A continuous count reflecting cumulative infectious load.
- **Time-to-Recovery**: Days until the first day with no illness AND no diarrhea, capped at 60 days. Measures how quickly a child clears infection.
- **Diarrhea Risk** (classification, V1 only): Binary — will the child experience any diarrhea in the next window? Ultimately deprioritized (see Section 6.4).

Additionally, **BAZ Autoregressive** (predicting the next Z-score level) was evaluated as a V1 static target.

### 1.2 Model Types Evaluated

| Tier | Models | Input | Architecture |
|------|--------|-------|-------------|
| Baseline | Naive heuristics | — | Zero-growth, persistence, last-interval |
| V1 (Static) | Random Forest, XGBoost, CatBoost | ~11-12 clinical features per visit | Tree ensembles with GroupKFold CV |
| V2 (Sequence) | LSTM, TCN | 5 visits × 23 features (windowed) | Deep temporal networks |
| V4 (Hybrid) | Hybrid LSTM, Hybrid TCN | 64-dim embeddings + V1 features | XGBoost on fused representation |

### 1.3 Evaluation Strategy

- **GroupKFold** (5-fold for V1, GroupShuffleSplit 80/20 for V2/V4) with `groups=pid`. Every observation from a given child appears in either train or test, never both.
- **Metrics**: R² (explained variance), MAE (average absolute error), RMSE (root mean squared error).
- **Clinical priority**: MAE is the most clinically meaningful metric because it directly measures "how wrong is this prediction in natural units?"

---

## 2. BASELINE PERFORMANCE ANALYSIS

### 2.1 Baseline Definitions

Each target has a carefully chosen naive baseline that represents the simplest possible prediction strategy:

| Target | Baseline Strategy | Rationale |
|--------|------------------|-----------|
| ΔBAZ | **Zero Growth** (predict 0) | Assumes the child's BAZ won't change — the "nothing happens" hypothesis |
| Illness Burden | **Persistence** (last window's illness count) | Assumes illness continues at the same rate — the "momentum" hypothesis |
| Recovery | **Last Interval** (previous inter-visit gap in days) | Assumes recovery takes as long as the last visit gap |

### 2.2 Baseline Scores

| Target | Baseline R² | Baseline MAE | Baseline RMSE |
|--------|-------------|-------------|---------------|
| ΔBAZ | **-0.002** | 0.285 | 0.430 |
| Illness | **0.128** | 2.149 | 5.377 |
| Recovery | **-2.160** | 31.310 | 79.621 |

### 2.3 What These Baselines Reveal

**ΔBAZ baseline R² ≈ 0**: The zero-growth baseline achieves R² near zero, which means predicting "no change" is essentially equivalent to predicting the population mean. This tells us that growth velocity has no strong directional bias — children are roughly equally likely to improve or deteriorate. The baseline MAE of 0.285 Z-score units sets the floor: any model must beat this to add value.

**Illness baseline R² = 0.128**: The persistence baseline explains 12.8% of variance, indicating that illness burden has *some* temporal autocorrelation — children who were recently sick tend to stay sick. But 87% of variance is unexplained, meaning illness patterns are far from deterministic. The persistence assumption captures the "momentum" of chronic illness but misses recovery events, acute onset, and seasonal variation.

**Recovery baseline R² = -2.16**: This profoundly negative R² means the last-interval baseline is *catastrophically worse than predicting the mean*. Recovery time is highly variable and bears almost no relationship to the previous visit gap. This is actually encouraging — it means any model that achieves positive R² is capturing genuine predictive signal, not just exploiting simple autocorrelation.

**Why baselines matter**: Without baselines, a model achieving R² = 0.51 on recovery looks mediocre. With baselines, we see it represents a leap from R² = -2.16 to +0.51 — a transformation from worse-than-useless to genuinely predictive. Baselines prevent the illusion of poor performance when the problem itself is inherently difficult.

---

## 3. V1 MODEL PERFORMANCE (TABULAR)

### 3.1 BAZ Autoregressive

| Model | R² | MAE | RMSE |
|-------|-----|-----|------|
| **CatBoost** | **0.843** | **0.298** | **0.448** |
| XGBoost | 0.838 | 0.302 | 0.456 |
| Random Forest | 0.838 | 0.306 | 0.456 |

**Interpretation**: All three models perform within 0.5% R² of each other, indicating the signal is fully extractable by any competent tree ensemble. The dominant feature is `target_prev` (SHAP = 0.80), meaning 80% of the prediction is "the child's next BAZ will be close to their current BAZ." Remaining signal comes from `Weight (kg)` (SHAP = 0.059), `target_velocity` (0.054), and `agedays` (0.035).

**What this means**: BAZ prediction is largely an autocorrelation problem. The model adds ~4 percentage points of R² beyond simple inertia, primarily by incorporating illness burden and recovery state. Clinically, an MAE of 0.298 means predictions are off by less than one-third of a Z-score unit — sufficient for identifying children trending toward wasting (Z < -2).

### 3.2 Growth Velocity (ΔBAZ) — V1

| Model | R² | MAE | RMSE |
|-------|-----|-----|------|
| **CatBoost** | **0.079** | **0.293** | **0.444** |
| Random Forest | 0.057 | 0.299 | 0.449 |
| XGBoost | 0.042 | 0.299 | 0.453 |

**Interpretation**: Static models explain less than 8% of ΔBAZ variance. The top SHAP features are `Height (cm)` (0.096), `Weight (kg)` (0.095), and `target_velocity` (0.065) — physical measurements rather than illness features. This means single-visit clinical features carry very little information about *how much* a child's Z-score will change.

**Why this happens**: Growth velocity is driven by events that unfold *between* visits — infections, dietary changes, catch-up growth after illness. A single snapshot at time t cannot capture these dynamics. The MAE barely improves over baseline (0.293 vs 0.285), confirming that static tabular models add minimal value for this target.

### 3.3 Illness Burden — V1

| Model | R² | MAE (days) | RMSE |
|-------|-----|-----------|------|
| **CatBoost** | **0.640** | **2.110** | **4.314** |
| XGBoost | 0.629 | 2.116 | 4.378 |
| Random Forest | 0.626 | 2.144 | 4.396 |

**Interpretation**: Illness burden is the strongest V1 target with R² = 0.64. The top SHAP features are `burden_illness_30d` (1.95), `Fever, caregiver report` (1.14), `Any illness, caregiver report` (0.93), and `agedays` (0.80). The dominant predictor is recent illness burden — children who have been frequently sick in the past 30 days are predicted to remain sick.

**Why this works**: Infectious disease in children follows a self-reinforcing cycle: repeated gut infections → damaged intestinal barrier → impaired nutrient absorption → weakened immunity → more infections. The burden features directly capture this cycle. The model predicts illness days with an average error of 2.1 days, clinically useful for resource allocation.

### 3.4 Strengths and Weaknesses of V1 Models

**Strengths**:
- Fast training and inference (suitable for real-time clinical tools).
- Highly interpretable via SHAP (clinicians can see *why* a child was flagged).
- Strong on targets with direct feature correlates (BAZ ← target_prev, illness ← burden_30d).
- CatBoost's ordered boosting provides slight but consistent advantage over XGBoost/RF.

**Weaknesses**:
- Cannot capture temporal trajectories (a child declining over 3 visits looks the same as a stable child at the same level).
- Limited to features available at a single visit.
- Poor on targets driven by inter-visit dynamics (ΔBAZ: R² = 0.079).

---

## 4. SEQUENCE MODEL PERFORMANCE (LSTM vs TCN)

### 4.1 Metrics

| Target | LSTM R² | LSTM MAE | TCN R² | TCN MAE |
|--------|---------|----------|--------|---------|
| ΔBAZ | 0.155 | 0.267 | 0.153 | 0.268 |
| Illness | 0.538 | 1.701 | 0.529 | 1.768 |
| Recovery | 0.512 | 12.075 | 0.489 | 13.638 |

### 4.2 LSTM vs TCN: Why They Perform Similarly

The LSTM and TCN achieve virtually identical results across all targets (R² difference < 0.03 everywhere). This is a meaningful finding, not a trivial one.

**What it tells us about the temporal dependencies**: The temporal patterns in this data are *short-range and relatively simple*. With a window of only 5 visits, both architectures can fully observe the entire sequence. The LSTM's theoretical advantage — its ability to maintain long-range memory through gating — is irrelevant when the sequence length is 5. The TCN's dilated convolutions with receptive field [1, 2] easily cover 5 timesteps.

**Implication**: The relevant temporal signal in pediatric health outcomes spans approximately 2-3 consecutive visits (2-3 months). A child's declining trajectory over the last 3 visits is predictive; their status from 10 visits ago is not. This aligns with clinical literature on acute malnutrition recovery windows (4-8 weeks).

**LSTM slight edge on recovery**: LSTM achieves MAE = 12.075 vs TCN's 13.638 on recovery — the largest gap across targets. Recovery prediction may benefit from LSTM's ordered processing (left-to-right through the sequence), which naturally models "how long has this child been sick" as a cumulative state. TCN processes all positions simultaneously and may not capture this sequential accumulation as naturally.

### 4.3 What Temporal Patterns Were Captured

Sequence models improve on V1 most dramatically for ΔBAZ:
- V1 R² = 0.126 → LSTM R² = 0.155 (23% relative improvement)
- V1 MAE = 0.273 → LSTM MAE = 0.267 (2.2% reduction)

The improvement is modest in absolute terms but reveals that *some* growth velocity signal lives in the temporal trajectory that static features cannot capture. For illness and recovery, sequence models perform comparably to V1 — indicating that the tabular features (burden_30d, recovery_days_since) already encode the temporal information that LSTM/TCN would learn.

---

## 5. HYBRID MODEL PERFORMANCE

### 5.1 Full Comparison

#### ΔBAZ (Growth Velocity)

| Model | R² | MAE | RMSE | MAE vs V1 |
|-------|-----|-----|------|-----------|
| Baseline (Zero Growth) | -0.002 | 0.285 | 0.430 | — |
| V1 (XGBoost Tabular) | 0.126 | 0.273 | 0.402 | baseline |
| LSTM | 0.155 | 0.267 | 0.395 | -2.2% |
| TCN | 0.153 | 0.268 | 0.396 | -1.8% |
| **Hybrid LSTM** | **0.546** | **0.188** | **0.290** | **-31.1%** |
| Hybrid TCN | 0.475 | 0.206 | 0.311 | -24.5% |

#### Illness Burden

| Model | R² | MAE (days) | RMSE | MAE vs V1 |
|-------|-----|-----------|------|-----------|
| Baseline (Persistence) | 0.128 | 2.149 | 5.377 | — |
| V1 (XGBoost Tabular) | 0.530 | 1.675 | 3.947 | baseline |
| LSTM | 0.538 | 1.701 | 3.917 | +1.6% |
| TCN | 0.529 | 1.768 | 3.953 | +5.6% |
| **Hybrid LSTM** | **0.609** | **1.487** | **3.601** | **-11.2%** |
| Hybrid TCN | 0.602 | 1.543 | 3.631 | -7.9% |

#### Recovery (Time-to-Recovery)

| Model | R² | MAE (days) | RMSE | MAE vs V1 |
|-------|-----|-----------|------|-----------|
| Baseline (Last Interval) | -2.160 | 31.310 | 79.621 | — |
| V1 (XGBoost Tabular) | 0.516 | 12.947 | 31.171 | baseline |
| LSTM | 0.512 | 12.075 | 31.297 | -6.7% |
| TCN | 0.489 | 13.638 | 32.012 | +5.3% |
| **Hybrid LSTM** | **0.511** | **11.687** | **31.320** | **-9.7%** |
| Hybrid TCN | 0.455 | 12.938 | 33.070 | -0.1% |

### 5.2 Why Hybrid Works Better

The hybrid model's architecture — concatenating 64-dimensional LSTM embeddings with ~12 V1 clinical features, then training XGBoost on the combined ~76-dimensional space — consistently outperforms both pure approaches. The mechanism is **information fusion**:

**What the LSTM embedding encodes**: The embedding is a 64-dimensional compressed summary of the child's *trajectory* over 5 visits. It encodes patterns like "BAZ has been declining for 3 consecutive visits while illness rate has been increasing" — multi-step temporal dynamics that no single-visit feature captures.

**What the V1 features encode**: Current clinical state — exact weight, exact burden count, exact days-since-recovery. These provide *precision* that the compressed embedding cannot.

**Why XGBoost on the combined space excels**: XGBoost discovers non-linear interactions between temporal dynamics and current state. For example: "LSTM embedding suggests declining trajectory (embedding dimension 17 is negative) AND current burden_illness_30d > 10 → predict high illness burden next month." Neither the embedding alone (which doesn't know the exact burden count) nor the V1 features alone (which don't know the trajectory) could make this prediction.

### 5.3 The ΔBAZ Breakthrough

The most dramatic hybrid improvement is on ΔBAZ: R² jumps from 0.155 (LSTM alone) to 0.546 — a **3.5× improvement**. MAE drops from 0.267 to 0.188, a 31% reduction.

**Why this is the strongest hybrid signal**: For ΔBAZ, V1 features are weak predictors (R² = 0.079 with static V1 CatBoost, 0.126 with V1 XGBoost on sequence data) and LSTM alone is modest (R² = 0.155). But when combined, R² reaches 0.546. This suggests that growth velocity depends on a *specific combination* of trajectory and context that neither source alone contains:
- The trajectory tells us the child is declining.
- The clinical features tell us the child is currently sick and hasn't recovered from the last infection.
- Together, the model predicts a continued decline — correctly.

This is the project's strongest empirical finding and the most compelling argument for the hybrid architecture.

---

## 6. TARGET-WISE ANALYSIS

### 6.1 🔵 ΔBAZ — The Most Challenging and Most Revealing Target

**Performance ceiling**: Even the best hybrid model explains only 54.6% of ΔBAZ variance. Nearly half of growth changes remain unpredictable.

**Why ΔBAZ is inherently noisy**:
1. **Measurement error**: Small errors in weight (±100g) or height (±0.5cm) produce disproportionate errors in Z-score changes, because Z-scores are standardized by narrow age-specific reference ranges.
2. **High-frequency dynamics**: Growth velocity changes day-to-day in response to food intake, hydration status, and acute infections — processes not captured by monthly visit intervals.
3. **Biological stochasticity**: Even children with identical clinical profiles show different growth responses to illness, reflecting unmeasured genetic variation in immune response and metabolic efficiency.

**What this reveals about biological growth**: The fact that static features predict only 8% of ΔBAZ but the hybrid captures 55% suggests that *growth changes are driven by temporal dynamics, not current state*. A child's growth trajectory over the past 3 months contains more information about their next growth change than any set of measurements taken today. This has clinical implications: screening programs that rely on single-visit Z-score cutoffs will miss children who are actively deteriorating but haven't yet crossed the threshold.

**Error analysis**: The hybrid model's error distribution on ΔBAZ is well-behaved: median error = 0.12, p90 = 0.39, p99 = 0.81, max = 0.92 Z-score units. The largest errors occur at the distribution tails — children with extreme growth spurts or severe acute declines that are inherently hard to predict.

### 6.2 🟢 Illness Burden — The Strongest Temporal Target

**Performance**: Hybrid LSTM achieves R² = 0.609, MAE = 1.487 illness days. The model predicts how many days a child will be sick in the next month with an average error of 1.5 days.

**Why illness is temporally structured**: Infectious disease in the MAL-ED cohort follows a vicious cycle documented extensively in the clinical literature:
1. Enteric infection → intestinal barrier damage → environmental enteropathy.
2. Damaged gut → impaired nutrient absorption → micronutrient deficiency.
3. Micronutrient deficiency → weakened immune response → susceptibility to next infection.
4. Cycle repeats with compounding severity.

This cycle creates strong temporal autocorrelation: a child's illness burden in month N is highly predictive of their burden in month N+1. The burden features (`burden_illness_30d`, SHAP = 1.95) directly capture this, which is why even V1 achieves R² = 0.640 on the static dataset.

**Why the hybrid adds value**: The hybrid improves R² from 0.530 (V1 on sequence data) to 0.609 (+14.9% relative). The LSTM embedding captures *patterns in how illness burden has been changing*. A child whose illness burden has been steadily increasing over 5 visits (even if the current count is moderate) is at higher risk than one with a single high-burden visit. The hybrid model detects this trajectory.

**Error analysis**: Median error = 0.36 days. p90 = 4.75 days. p99 = 13.50 days. Max error = 31.52 days. The heavy tail (max >> p90) occurs because illness burden has a highly skewed distribution — most windows have 0 illness days (median target = 0), but some children experience prolonged 20-30 day illness episodes that are difficult to predict.

### 6.3 🟣 Time-to-Recovery — Moderate Predictability

**Performance**: Hybrid LSTM achieves R² = 0.511, MAE = 11.687 days.

**Why recovery is partially predictable**: Recovery time depends on:
- *Current illness severity* (captured by V1 features): children with higher current burden take longer to recover.
- *Immune resilience* (partially captured by trajectory): children who have recovered quickly in the past tend to recover quickly again.
- *Age* (SHAP = 0.80 for illness_burden, suggesting age-related immunity maturation).

**Why hybrid improves MAE but not R²**: The hybrid achieves the best MAE (11.687 vs V1's 12.947, a 9.7% improvement) but its R² (0.511) is *slightly lower* than V1's (0.516). This metric disagreement tells us something important: the hybrid reduces average prediction error (MAE) but introduces slightly more variance in its predictions (inflating RMSE, which depresses R²). The clinical interpretation: the hybrid makes better predictions *on average* but is occasionally more wrong on outlier cases. Since MAE is the clinically relevant metric ("how far off is the prediction for a typical patient?"), the hybrid is preferred.

**What limits recovery prediction**: Recovery time has a peculiar distribution — many children recover within 1 day (first clear day after the visit), while others take 30-60+ days. This bimodal structure means the model must simultaneously predict "fast recovery" and "prolonged illness" cases, which have very different feature signatures. The error analysis confirms this: p90 = 26.3 days, p99 = 87.0 days, max = 125.7 days. The largest errors occur for children who take unexpectedly long to recover, likely due to unmeasured factors (secondary infections, treatment adherence, dietary quality).

### 6.4 🟡 Diarrhea Risk — Why It Was Removed

**Initial appeal**: Diarrhea is the leading cause of child mortality in MAL-ED settings, making it a natural prediction target. CatBoost achieved ROC-AUC = 0.873, suggesting strong discriminative ability.

**Why the performance was misleading**:

1. **Extreme class imbalance**: Only 6.2% of observations had diarrhea (1,487 positive out of 24,052). The baseline (DummyClassifier with `strategy="prior"`) achieves F1 = 0.908 and F2 = 0.926 by simply predicting "no diarrhea" most of the time. CatBoost's F1 = 0.830 is actually *lower* than this baseline — the model sacrificed overall F1 to improve AUC by detecting some true positives.

2. **Circular prediction**: The top predictor is `burden_diarrhea_30d` — "the child had diarrhea recently, so they'll have diarrhea again." While technically accurate, this is not clinically actionable. A health worker already knows the child had diarrhea recently; they need to know about children who *haven't* had diarrhea yet but are about to.

3. **The safe variant test**: A "diarrhea-safe" feature set was created that excluded diarrhea-history features (`burden_diarrhea_30d`, `recovery_days_since_diarrhea`, `Diarrhea, caregiver report`). Without these, performance dropped significantly, confirming that the model was learning a trivial pattern rather than genuine clinical risk factors.

4. **Why removed from temporal modeling**: The binary target loses granularity — knowing that "the child will have diarrhea" is less useful than knowing "the child will have 8 illness days" (illness burden already captures diarrhea severity as a component). The temporal pipeline (V2/V4) excluded diarrhea as a standalone target because illness burden provides strictly more information.

---

## 7. MODEL COMPARISON INSIGHTS

### 7.1 No Single Model Dominates Everywhere

| Metric Winner | ΔBAZ | Illness | Recovery |
|--------------|------|---------|----------|
| **Best R²** | Hybrid LSTM (0.546) | Hybrid LSTM (0.609) | V1 XGBoost (0.516) |
| **Best MAE** | Hybrid LSTM (0.188) | Hybrid LSTM (1.487) | Hybrid LSTM (11.687) |

The Hybrid LSTM wins on MAE everywhere but loses to V1 on R² for recovery. This metric disagreement (discussed in Section 6.3) illustrates why multi-metric evaluation is essential.

### 7.2 When to Use Which Model

- **V1 CatBoost**: When interpretability is paramount. SHAP explanations are available for every prediction. Best for BAZ autoregressive (R² = 0.843, full SHAP suite) where the goal is screening rather than precise forecasting.
- **LSTM alone**: Limited practical use. It is outperformed by the hybrid in every scenario. Its value is as a component of the hybrid pipeline, not as a standalone model.
- **Hybrid LSTM**: Best overall predictive accuracy. Use when the goal is maximum prediction quality and the 64-dim embedding's opacity is acceptable. The clinical trade-off: better predictions but harder to explain *why*.

### 7.3 The Diminishing Returns Pattern

The improvement from each tier shows diminishing returns for some targets:

| Target | Baseline → V1 | V1 → LSTM | LSTM → Hybrid |
|--------|---------------|-----------|---------------|
| ΔBAZ | +0.128 R² | +0.029 R² | **+0.391 R²** |
| Illness | +0.402 R² | +0.008 R² | **+0.071 R²** |
| Recovery | +2.676 R² | -0.004 R² | -0.001 R² |

ΔBAZ breaks the pattern — the hybrid provides an *acceleration* of improvement, not a diminishment. For illness, the returns diminish but remain positive. For recovery, the sequence/hybrid tier adds no R² improvement (though MAE still improves), suggesting that recovery is fundamentally a state-driven rather than trajectory-driven target.

---

## 8. ERROR ANALYSIS

### 8.1 Error Distribution Summary (V1 CatBoost, Static)

| Target | Median Error | p90 Error | p95 Error | p99 Error | Max Error |
|--------|-------------|-----------|-----------|-----------|-----------|
| BAZ AR | 0.211 | 0.633 | 0.840 | 1.597 | 6.058 |
| ΔBAZ | 0.206 | 0.619 | 0.832 | 1.547 | 6.010 |
| Illness | 0.363 | 6.661 | 10.119 | 17.125 | 118.246 |

### 8.2 Error Distribution Summary (V2/V4 Temporal, from 200-sample prediction sets)

| Target / Model | MAE | p90 Error | p99 Error | Max Error |
|----------------|-----|-----------|-----------|-----------|
| ΔBAZ / Hybrid LSTM | **0.173** | 0.390 | 0.815 | 0.922 |
| ΔBAZ / LSTM | 0.256 | 0.550 | 1.053 | 1.719 |
| ΔBAZ / V1 XGBoost | 0.251 | 0.574 | 1.447 | 1.548 |
| Illness / Hybrid LSTM | **1.334** | 4.746 | 13.497 | 31.521 |
| Illness / LSTM | 2.034 | 6.673 | 17.592 | 24.257 |
| Illness / V1 XGBoost | 1.710 | 5.677 | 18.795 | 22.493 |
| Recovery / Hybrid LSTM | **10.470** | 26.322 | 87.024 | 125.655 |
| Recovery / LSTM | 9.851 | 25.109 | 84.827 | 154.867 |
| Recovery / V1 XGBoost | 12.798 | 31.344 | 94.701 | 174.029 |

### 8.3 Where Models Fail

**High-error cases in ΔBAZ**: The largest errors (p99 = 1.55 for V1, 0.81 for Hybrid) occur at the distribution extremes — children with sudden, large Z-score changes (|ΔBAZ| > 1.0). These correspond to acute events: severe diarrheal episodes causing rapid weight loss, or catch-up growth after recovery. Since the model has no direct feature for "severity of the current acute event," it underestimates the magnitude of these extreme changes. The hybrid's p99 (0.81) is nearly half of V1's (1.55), demonstrating that the temporal trajectory provides advance warning of extreme changes.

**Heavy tail in illness burden**: The illness error distribution has a massive right tail (max = 118 for V1, 31.5 for Hybrid). This occurs because illness burden has a zero-inflated distribution — most children have 0 illness days in a given window, but a minority experience prolonged 20-145 day illness episodes. The model effectively predicts the "zero" majority well (low median error = 0.36) but struggles with the "prolonged illness" minority because these extreme events depend on pathogen-specific factors not in the feature set.

**Recovery prediction extremes**: Recovery max errors exceed 100 days across all models. These correspond to children with extremely prolonged illness (recovery capped at 60 days in the target definition, but some observations exceed this cap in the raw data with values up to 910 days). The model has no way to distinguish a child who will recover in 5 days from one who will develop chronic environmental enteropathy lasting months.

### 8.4 Bias vs Variance Analysis

**ΔBAZ models show variance reduction**: The hybrid's p99 error (0.81) is dramatically lower than LSTM's (1.05) and V1's (1.45). The hybrid doesn't just improve the mean — it compresses the error distribution, reducing extreme failures. This suggests the XGBoost on fused features acts as a regularizer, preventing the wild predictions that pure sequence models occasionally produce.

**Illness models show bias reduction**: The hybrid's MAE improvement (1.33 vs V1's 1.71) is a bias reduction — the model's central tendency is more accurate. But its max error (31.5) exceeds LSTM's max (24.3), suggesting occasional high-variance outlier predictions when the embedding contains unusual patterns. The clinical implication: the hybrid is better on average but should be supplemented with prediction intervals for high-stakes decisions.

**Recovery models show mixed behavior**: The hybrid achieves the best MAE (10.47) but LSTM achieves the best p90 (25.1) and max (154.9 vs 125.7). The hybrid occasionally produces very large errors for edge cases. This is likely because the XGBoost on embeddings can extrapolate more aggressively than the LSTM's bounded output.

---

## 9. TEMPORAL VS TABULAR LEARNING INSIGHTS

### 9.1 What Tabular Models Learned

V1 tree models learned **direct feature-to-target mappings**:
- BAZ AR: "Next BAZ ≈ current BAZ + small correction for weight and illness burden"
- Illness: "Next illness ≈ f(recent_illness_count, current_fever_status, age)"
- ΔBAZ: Almost nothing useful (R² = 0.079) — the tabular features are insufficient.

The top SHAP features for each target confirm that V1 models rely on the most obvious clinical correlates. They cannot discover multi-step patterns like "this child has been declining for 3 months" because each training sample contains only one visit's features.

### 9.2 What Sequence Models Learned

LSTM and TCN models learned **temporal dynamics**:
- Short-term trends: "BAZ has been declining for 2-3 visits" → predict further decline.
- Recovery patterns: "Illness burden peaked 2 visits ago and is now decreasing" → predict shorter recovery.
- Phase transitions: "Child transitioned from infant to toddler phase AND growth slowed" → predict growth velocity change.

The key evidence: sequence models improve most on ΔBAZ (the target with the weakest static signal), demonstrating that temporal patterns contain information that single-visit features fundamentally cannot.

### 9.3 Why Combining Both Is Powerful

The hybrid architecture succeeds because temporal dynamics and clinical context are *complementary*, not redundant information sources:

| Information Type | Source | Example |
|-----------------|--------|---------|
| **Trajectory direction** | LSTM embedding | "This child has been declining for 3 visits" |
| **Trajectory speed** | LSTM embedding | "The decline is accelerating" |
| **Current absolute state** | V1 features | "Current weight = 8.2 kg, burden = 12 illness days" |
| **Recovery phase** | V1 features | "Last diarrhea was 45 days ago" |
| **Socioeconomic context** | V1 features | "WAMI index = 0.3 (low)" |

The XGBoost learns conditional rules that span both sources:
- "IF trajectory is declining AND current weight is below age-median AND WAMI is low → predict severe outcome"
- "IF trajectory was declining but has inflected upward AND recovery_days_since_illness > 30 → predict improvement"

These compound rules explain the super-additive improvement on ΔBAZ: V1 R² = 0.126, LSTM R² = 0.155, but Hybrid R² = 0.546. The information gain from combining is far larger than the sum of individual gains.

---

## 10. KEY TAKEAWAYS

### 10.1 Data Insights

**Which phenomena are predictable**:
- **Illness burden** is the most predictable temporal outcome (R² = 0.609). The vicious cycle of enteric infection creates strong autocorrelation and temporal structure that models can exploit.
- **BAZ autoregressive** is highly predictable (R² = 0.843) but this is mostly inertia — the real clinical signal (the additional 4% from illness features) is small.
- **Recovery time** is moderately predictable (R² ≈ 0.51). Current illness severity drives most of the prediction; trajectory adds modest improvement.

**Which are inherently noisy**:
- **Growth velocity (ΔBAZ)** has a hard ceiling around R² ≈ 0.55 even with the best models. Nearly half of growth changes are driven by unmeasured factors (dietary intake, pathogen virulence, genetic variation in immune response).
- The noise is not random — it reflects genuine biological complexity that would require different data sources (dietary logs, microbiome composition, specific pathogen identification) to capture.

### 10.2 Model Insights

**When LSTM helps**:
- When the target depends on *trajectory* rather than *state* (ΔBAZ: LSTM adds +0.03 R² over V1; hybrid adds +0.42).
- When temporal dynamics are non-obvious (declining over multiple visits, cyclical illness patterns).
- When used as a feature extractor for hybrid models, not as a standalone predictor.

**When LSTM doesn't help**:
- When the target is primarily state-dependent (recovery: LSTM R² = 0.512 ≈ V1 R² = 0.516).
- When V1 features already encode temporal information (burden_30d, recovery_days_since already summarize recent history).
- As a standalone model — the LSTM alone is never the best option for any target.

**Why hybrid works best**:
- It combines two orthogonal information sources: compressed temporal dynamics (64-dim embedding) + precise current state (12 clinical features).
- XGBoost discovers interaction effects between trajectory and context that neither source reveals alone.
- The most dramatic improvement occurs on the hardest target (ΔBAZ), where the interaction between trajectory and context is most informative.

### 10.3 Clinical Insights

**What these predictions mean in practice**:

- **Illness burden (MAE = 1.5 days)**: A clinician can predict whether a child will have a light month (0-2 illness days) vs a heavy month (10+ illness days) with reasonable accuracy. This enables proactive resource allocation — scheduling more frequent check-ups for high-risk children.

- **Recovery time (MAE = 11.7 days)**: Predictions are accurate to within ~12 days. A prediction of "recovery in 5 days" vs "recovery in 30 days" is distinguishable, enabling different treatment protocols (outpatient vs. inpatient monitoring).

- **Growth velocity (MAE = 0.19 Z-score units)**: The model can detect children who will lose > 0.3 Z-score units in the next period — sufficient for early warning before they cross the wasting threshold. The clinical value is in *identifying declining trajectories before they become emergencies*.

- **The trajectory matters more than the snapshot**: The hybrid model's success on ΔBAZ demonstrates that a child's *history* contains critical information beyond their current measurements. Clinical screening programs should incorporate multi-visit trajectory analysis, not just single-visit Z-score cutoffs.

---

## 11. LIMITATIONS OF RESULTS

### 11.1 Weak Targets

ΔBAZ remains 45% unexplained. The residual variance is likely driven by unmeasured factors — dietary intake, specific pathogen types, household hygiene practices, and genetic variation. Without these data, there is a hard ceiling on ΔBAZ prediction accuracy. The hybrid model may be approaching this ceiling.

### 11.2 Data Sparsity

With only ~5-10 anthropometric measurements per child per year, the temporal models work with very short sequences (window=5). Children with fewer than 6 visits are excluded entirely, creating a survivorship bias toward children with better healthcare access. The LSTM/TCN architectures are likely underfitting due to limited training samples (18,528 sequences across 1,298 children).

### 11.3 Baseline Sensitivity

The V1 XGBoost baseline in the temporal evaluation was trained on flattened sequence features (5×23=115 features), not on the V1 clinical subset. This means it had access to more information than the true V1 model, making the V1 row in the temporal comparison tables slightly inflated. The "true" V1 static performance (from the V1 registry) uses only 11-12 features and achieves lower R² on ΔBAZ (0.079 vs 0.126).

### 11.4 Single Cohort

All results are from the MAL-ED cohort. Generalization to populations with different pathogen profiles, dietary patterns, or healthcare systems is unvalidated. The temporal patterns learned (illness cycling, growth recovery dynamics) may be cohort-specific.

### 11.5 No Uncertainty Quantification

All models produce point predictions. A prediction of "recovery in 15 days" would be more useful with a confidence interval (e.g., "15 ± 8 days, 90% CI"). Without uncertainty estimation, clinicians cannot assess how much to trust any individual prediction.

---

## 12. FUTURE DIRECTIONS (RESULT-DRIVEN)

### 12.1 Better Targets

Based on the results, several alternative target formulations could be more productive:

- **Composite risk score**: Instead of predicting ΔBAZ, illness, and recovery separately, create a single multi-dimensional risk score that combines all three. This could be more clinically actionable than individual predictions.
- **Stunting prediction (HAZ)**: Height-for-age Z-score changes more slowly than BMI and may be more predictable. Stunting is the primary indicator of chronic malnutrition.
- **Threshold crossing**: Instead of predicting the value of ΔBAZ, predict the binary outcome "will BAZ cross below -2?" This converts the noisy regression problem into a potentially easier classification with direct clinical meaning.
- **Multi-step forecasting**: Instead of predicting one step ahead, predict the child's status 3 or 6 months out. Longer horizons might be more useful for public health planning.

### 12.2 Alternative Formulations

- **Attention mechanisms**: Replacing LSTM with a Transformer could provide both better performance (via multi-head attention over the full sequence) and interpretability (attention weights reveal which past visits matter most).
- **Multi-task learning**: Training a single model to predict all targets simultaneously, sharing the temporal encoder. This could improve sample efficiency and reveal shared temporal representations.
- **Conformal prediction**: Adding prediction intervals via conformal methods would provide uncertainty quantification without changing the underlying model.

### 12.3 Improved Modeling

- **Longer sequences**: If data permits, extending from 5 to 10+ visit windows could capture longer-term patterns.
- **External data fusion**: Incorporating weather data (monsoon seasons), local disease surveillance reports, or dietary surveys could address the unmeasured factors limiting ΔBAZ prediction.
- **Ensemble of hybrids**: Combining Hybrid LSTM and Hybrid TCN predictions via stacking could further reduce variance.
- **Clinical validation**: Prospective deployment with a randomized controlled trial comparing model-assisted vs standard care would quantify the real-world impact of these predictions.

---

*End of Results & Insights Documentation*

*This document captures all experimental results, their interpretation, cross-target comparisons, error analysis, and the clinical implications of the MAL-ED Clinical Nexus modeling system. Every finding is traced from observed metric to biological explanation to clinical implication, providing comprehensive source material for the Results and Discussion sections of an academic report.*
