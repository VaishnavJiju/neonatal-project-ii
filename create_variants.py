import pandas as pd
import numpy as np
import hashlib
import os

# Base dataset path (relative to project root)
BASE_DATASET_PATH = "mal_ed_data/mal_ed_final.parquet"

# Output directory for processed variants
OUTPUT_DIR = "mal_ed_data/processed_variants"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Configuration for each target
# ---------------------------------------------------------------------------
TARGET_CONFIG = {
    "BAZ": {
        "target_col": "BMI-for-age z-score",
        "exclude": [
            "Weight-for-age z-score",
            "Length-for-age z-score",
            "Weight-for-length z-score",
        ],
    },
    "LAZ": {
        "target_col": "Length-for-age z-score",
        "exclude": [
            "BMI-for-age z-score",
            "Weight-for-age z-score",
            "Weight-for-length z-score",
        ],
    },
    "WAZ": {
        "target_col": "Weight-for-age z-score",
        "exclude": [
            "BMI-for-age z-score",
            "Length-for-age z-score",
            "Weight-for-length z-score",
        ],
    },
}

# ---------------------------------------------------------------------------
# Safe static and clinical feature sets (low‑risk, high‑signal)
# ---------------------------------------------------------------------------
SAFE_STATIC_FEATURES = [
    "WAMI index",
    "Maternal education",
    "Family income",
    "Household density",
]

SAFE_ILLNESS_FEATURES = [
    "Any illness, caregiver report",
    "Diarrhea, caregiver report",
    "Antibiotic usage",
]

SAFE_FEEDING_FEATURES = [
    "Breastfeeding status",
    "Active breastfeeding",
]

SAFE_FEATURES = SAFE_STATIC_FEATURES + SAFE_ILLNESS_FEATURES + SAFE_FEEDING_FEATURES

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------
def generate_dataset_id(df: pd.DataFrame) -> str:
    """Deterministic hash based on the first 1000 rows of the processed dataframe."""
    sample_csv = df.head(1000).to_csv(index=False).encode()
    return hashlib.sha256(sample_csv).hexdigest()

def convert_categorical(series: pd.Series) -> pd.Series:
    """Convert Yes/No to 1/0, Unknown to NaN, and attempt ordinal conversion.
    This is a very lightweight conversion; more complex mappings can be added.
    """
    if series.dtype == object:
        # Standard binary mapping
        mapping = {"Yes": 1, "No": 0, "yes": 1, "no": 0, "Y": 1, "N": 0}
        # Replace unknowns with NaN
        series = series.replace({"Unknown": np.nan, "unknown": np.nan, "": np.nan})
        # Apply binary mapping where possible
        series = series.map(mapping).fillna(series)
        # Try to coerce remaining strings to numeric (ordinal text)
        try:
            series = pd.to_numeric(series, errors="coerce")
        except Exception:
            pass
    return series

def preprocess_target(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    # ---------------------------------------------------------------------
    # Step 1: Basic validation & sorting
    # ---------------------------------------------------------------------
    required_cols = {"pid", "agedays", cfg["target_col"]}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns for target {cfg['target_col']}: {missing}")

    df = df.sort_values(["pid", "agedays"]).reset_index(drop=True)

    # ---------------------------------------------------------------------
    # Step 2: Column filtering – keep only safe features + target + ids
    # ---------------------------------------------------------------------
    # Start with mandatory columns
    keep_cols = {"pid", "agedays", cfg["target_col"]}
    # Add safe features that exist in the dataframe
    for col in SAFE_FEATURES:
        if col in df.columns:
            keep_cols.add(col)
    # Remove any explicitly excluded columns (other z‑scores, ratios, etc.)
    for excl in cfg["exclude"]:
        keep_cols.discard(excl)
    # Also drop any column that contains "z-score" other than the target
    for col in list(df.columns):
        if "z-score" in col.lower() and col != cfg["target_col"]:
            keep_cols.discard(col)
    # Drop anthropometric ratios if present
    ratio_keywords = ["weight-for-length", "wfl", "bmi"]
    for col in list(df.columns):
        if any(kw in col.lower() for kw in ratio_keywords) and col != cfg["target_col"]:
            keep_cols.discard(col)
    # Finally, select the columns that are both in keep_cols and actually present
    df = df[[c for c in df.columns if c in keep_cols]].copy()


    # ---------------------------------------------------------------------
    # Step 3: Categorical handling
    # ---------------------------------------------------------------------
    for col in df.select_dtypes(include=[object]).columns:
        if col == "pid":
            continue
        df[col] = convert_categorical(df[col])

    # ---------------------------------------------------------------------
    # Step 4: Missing value handling – forward fill within each child then drop any remaining NA
    # ---------------------------------------------------------------------
    # NOTE: Forward fill will be applied AFTER temporal features are created.
    # Placeholder – actual forward fill of stable columns is performed later.
    pass

    # ---------------------------------------------------------------------
    # Step 5: Add time feature
    # ---------------------------------------------------------------------
    df["age_months"] = df["agedays"] / 30.0

    # ---------------------------------------------------------------------
    # Step 6: Temporal features for the selected target
    # ---------------------------------------------------------------------
    target_col = cfg["target_col"]
    # Lag (previous assessment)
    df["target_prev"] = df.groupby("pid")[target_col].shift(1)
    # Shift forward to create the prediction target (t+1)
    df["target"] = df.groupby("pid")[target_col].shift(-1)
    # Velocity (rate of change per day)
    delta = df.groupby("pid")[target_col].diff()
    time_diff = df.groupby("pid")["agedays"].diff()
    df["target_velocity"] = (delta / time_diff)
    df["target_velocity"] = df["target_velocity"].replace([np.inf, -np.inf], 0)
    df["target_velocity"] = df["target_velocity"].fillna(0)
    df["target_velocity"] = df["target_velocity"].clip(-5, 5)

    # ---------------------------------------------------------------------
    # Step 7: Drop rows where any of the newly created target columns are missing
    # ---------------------------------------------------------------------
    df = df.dropna(subset=["target", "target_prev"])
    
    # Remove original target column to prevent data leakage
    df = df.drop(columns=[target_col])
    
    # ---------------------------------------------------------------------
    # Step 8: Forward fill stable columns only (after temporal features)
    # ---------------------------------------------------------------------
    stable_cols = [
        "WAMI index",
        "Maternal education",
        "Family income",
        "Household density",
    ]
    # Only forward fill columns that exist in the dataframe
    stable_cols = [c for c in stable_cols if c in df.columns]
    if stable_cols:
        df[stable_cols] = df.groupby("pid")[stable_cols].ffill()
    
    # ---------------------------------------------------------------------
    # Step 9: Final validation checks
    # ---------------------------------------------------------------------
    assert target_col not in df.columns
    assert not df.isna().any().any(), "Missing values remain after cleaning"
    assert df.sort_values(["pid", "agedays"]).equals(df), "Dataset not sorted after processing"

    # ---------------------------------------------------------------------
    # Step 8: Final validation checks
    # ---------------------------------------------------------------------
    # Ensure each child has at least 3 observations (enough for lag/lead)
    valid_pids = df.groupby("pid").size()
    valid_pids = valid_pids[valid_pids > 2].index
    df = df[df["pid"].isin(valid_pids)].copy()

    assert target_col not in df.columns
    assert not df.isna().any().any(), "Missing values remain after cleaning"
    assert df.sort_values(["pid", "agedays"]).equals(df), "Dataset not sorted after processing"

    return df

def save_variant(df: pd.DataFrame, variant_name: str):
    dataset_id = generate_dataset_id(df)
    filename = os.path.join(OUTPUT_DIR, f"processed_{variant_name}.parquet")
    df.to_parquet(filename, index=False)
    print(f"Saved {variant_name} variant ({df.shape[0]} rows, {df.shape[1]} cols) to {filename}")
    print(f"Dataset ID: {dataset_id}")

def main():
    # Load the raw dataset once
    raw_df = pd.read_parquet(BASE_DATASET_PATH)
    print(f"Loaded raw dataset with shape {raw_df.shape}")

    for variant, cfg in TARGET_CONFIG.items():
        print(f"\nProcessing variant: {variant} (target column: {cfg['target_col']})")
        processed = preprocess_target(raw_df, cfg)
        save_variant(processed, variant)

if __name__ == "__main__":
    main()
