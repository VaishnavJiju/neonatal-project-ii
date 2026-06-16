import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

DATA_PATH = "mal_ed_data/processed_variants/processed_BAZ.parquet"
df = pd.read_parquet(DATA_PATH)

# Clean up raw df for modeling
X_base = df.drop(columns=["pid", "target"]).copy()
# Map unknown categorical variables and enforce numeric features
for col in X_base.select_dtypes(include=[object]).columns:
    X_base[col] = X_base[col].replace({"Unknown": np.nan, "unknown": np.nan, "": np.nan})
    X_base[col] = pd.to_numeric(X_base[col], errors="coerce")
    median_val = X_base[col].median()
    if pd.isna(median_val):
        median_val = 0
    X_base[col] = X_base[col].fillna(median_val)

for col in X_base.select_dtypes(include=['category']).columns:
    X_base[col] = X_base[col].cat.codes

X_base = X_base.fillna(0)
y = df["target"]

print("========================================")
print("TEST 1 - Baseline Sanity")
print("========================================")
r2_baseline = r2_score(df["target"], df["target_prev"])
print(f"R² of just using target_prev: {r2_baseline:.4f}\n")


print("========================================")
print("TEST 2 - Kill-Switch Test")
print("========================================")
# Drop temporal inertial features
X_kill = X_base.drop(columns=["target_prev", "target_velocity"], errors="ignore")
X_train_k, X_test_k, y_train_k, y_test_k = train_test_split(X_kill, y, test_size=0.2, random_state=42)

rf_k = RandomForestRegressor(n_estimators=50, max_depth=15, max_samples=0.5, random_state=42, n_jobs=-1)
rf_k.fit(X_train_k, y_train_k)
pred_k = rf_k.predict(X_test_k)
r2_kill = r2_score(y_test_k, pred_k)
print(f"R² without target_prev/velocity: {r2_kill:.4f}\n")


print("========================================")
print("TEST 3 - Horizon Test (t+2)")
print("========================================")
# Create a new df to shift correctly within patients
df_hz = df.copy()
# Shift by -2 to get t+2
df_hz["target_t2"] = df_hz.groupby("pid")["target"].shift(-1) # target is already t+1, so shifting target by -1 gives t+2

# Drop the NaNs created by shifting at the end of patients' timelines
df_hz = df_hz.dropna(subset=["target_t2"])

y_hz = df_hz["target_t2"]
X_hz = df_hz.drop(columns=["pid", "target", "target_t2"]).copy()

# Apply the same categorical mapping for the horizon features
for col in X_hz.select_dtypes(include=[object]).columns:
    X_hz[col] = X_hz[col].replace({"Unknown": np.nan, "unknown": np.nan, "": np.nan})
    X_hz[col] = pd.to_numeric(X_hz[col], errors="coerce")
    median_val = X_hz[col].median()
    if pd.isna(median_val):
        median_val = 0
    X_hz[col] = X_hz[col].fillna(median_val)

for col in X_hz.select_dtypes(include=['category']).columns:
    X_hz[col] = X_hz[col].cat.codes

X_hz = X_hz.fillna(0)

X_train_h, X_test_h, y_train_h, y_test_h = train_test_split(X_hz, y_hz, test_size=0.2, random_state=42)

rf_h = RandomForestRegressor(n_estimators=50, max_depth=15, max_samples=0.5, random_state=42, n_jobs=-1)
rf_h.fit(X_train_h, y_train_h)
pred_h = rf_h.predict(X_test_h)
r2_hz = r2_score(y_test_h, pred_h)
print(f"R² predicting t+2 (with target_prev at t): {r2_hz:.4f}\n")
