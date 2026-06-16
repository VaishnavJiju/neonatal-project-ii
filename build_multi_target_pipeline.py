import pandas as pd
import numpy as np
import os
import gc

def binarize(series):
    series_str = series.astype(str).str.lower().str.strip()
    return series_str.isin(["yes", "1", "true"]).astype(int)

def build_pipeline():
    print("Loading Dataset...")
    df = pd.read_parquet("mal_ed_data/mal_ed_final.parquet")
    df = df.sort_values(["pid", "agedays"]).reset_index(drop=True)
    df["time_dt"] = pd.to_timedelta(df["agedays"], unit="D")
    
    # ---------------------------------------------------------
    # 1. TEMPORAL FEATURES (BURDEN & RECOVERY) ON DAILY DATA
    # ---------------------------------------------------------
    print("Engineering Burden and Recovery Matrices...")
    
    # Identify boolean event cols
    col_illness = "Any symptoms of illness, caregiver report" if "Any symptoms of illness, caregiver report" in df.columns else [c for c in df.columns if "any illness" in c.lower()][0]
    col_diar = "Diarrhea, caregiver report" if "Diarrhea, caregiver report" in df.columns else [c for c in df.columns if "diarrhea, caregiver report" in c.lower()][0]
    col_abx = "Use of antibiotics, caregiver report" if "Use of antibiotics, caregiver report" in df.columns else [c for c in df.columns if "antibiotic" in c.lower()][0]
    
    df["ev_illness"] = binarize(df[col_illness])
    df["ev_diarrhea"] = binarize(df[col_diar])
    df["ev_antibiotics"] = binarize(df[col_abx])
    
    # --- A. BURDEN (30-day rolling lookback, closed='left' to prevent leakage) ---
    df.set_index("time_dt", inplace=True)
    g = df.groupby("pid")
    
    df["burden_illness_30d"] = g["ev_illness"].rolling("30D", closed="left").sum().reset_index(level=0, drop=True)
    df["burden_diarrhea_30d"] = g["ev_diarrhea"].rolling("30D", closed="left").sum().reset_index(level=0, drop=True)
    df["burden_antibiotics_30d"] = g["ev_antibiotics"].rolling("30D", closed="left").sum().reset_index(level=0, drop=True)
    df.reset_index(inplace=True)
    
    df[["burden_illness_30d", "burden_diarrhea_30d", "burden_antibiotics_30d"]] = df[["burden_illness_30d", "burden_diarrhea_30d", "burden_antibiotics_30d"]].fillna(0)
    
    # --- B. RECOVERY (Days since last event, shifted by 1 to exclude today) ---
    for ev_col, out_col in [("ev_illness", "recovery_days_since_illness"), 
                            ("ev_diarrhea", "recovery_days_since_diarrhea"), 
                            ("ev_antibiotics", "recovery_days_since_antibiotics")]:
        # Day of the event
        event_days = df["agedays"].where(df[ev_col] == 1)
        # Forward fill the last event day per child, then shift 1 so "today" doesn't see "today's" event
        last_event_day = event_days.groupby(df["pid"]).ffill().groupby(df["pid"]).shift(1)
        df[out_col] = df["agedays"] - last_event_day
        df[out_col] = df[out_col].fillna(999) # 999 indicates "never happened"

    # ---------------------------------------------------------
    # 2. DEFINING THE TARGET ANCHORS (CLINICAL VISITS)
    # ---------------------------------------------------------
    print("Isolating Clinical Visits and Anchoring Targets...")
    target_col = "BMI-for-age z-score"
    df_visits = df.dropna(subset=[target_col]).copy()
    
    df_visits = df_visits.sort_values(["pid", "agedays"])
    
    # Next visit BAZ
    df_visits["target"] = df_visits.groupby("pid")[target_col].shift(-1)
    df_visits["target_prev"] = df_visits[target_col]
    df_visits["next_agedays"] = df_visits.groupby("pid")["agedays"].shift(-1)
    
    # Velocity
    delta = df_visits.groupby("pid")[target_col].diff()
    time_diff = df_visits.groupby("pid")["agedays"].diff()
    df_visits["target_velocity"] = (delta / time_diff)
    df_visits["target_velocity"] = df_visits["target_velocity"].replace([np.inf, -np.inf], 0).fillna(0).clip(-5, 5)
    
    df_visits = df_visits.dropna(subset=["target", "next_agedays", "target_prev"])
    
    # ---------------------------------------------------------
    # 3. FUTURE ILLNESS/DIARRHEA EXTRACTION (Target 3 & 4)
    # ---------------------------------------------------------
    print("Extracting Next-Window Illness Horizons...")
    diar_between = []
    illness_between = []
    
    # We loop over visits to sum up events that occur between the [current visit, next visit)
    # This takes 20-30 seconds but is perfectly leak-proof
    df_indexed = df.set_index(["pid", "agedays"])
    
    pids = df_visits["pid"].values
    t1 = df_visits["agedays"].values
    t2 = df_visits["next_agedays"].values
    
    for p, start_day, end_day in zip(pids, t1, t2):
        # We want events strictly > start_day and <= end_day
        try:
            subset = df_indexed.loc[p]
            mask = (subset.index > start_day) & (subset.index <= end_day)
            slice_df = subset.loc[mask]
            diar_between.append(slice_df["ev_diarrhea"].sum())
            illness_between.append(slice_df["ev_illness"].sum())
        except KeyError:
            diar_between.append(0)
            illness_between.append(0)
            
    df_visits["future_diarrhea_count"] = diar_between
    df_visits["future_illness_count"] = illness_between
    df_visits["target_diarrhea_bin"] = (df_visits["future_diarrhea_count"] > 0).astype(int)
    
    # ---------------------------------------------------------
    # 4. DATASET FACTORY (Outputs)
    # ---------------------------------------------------------
    print("Packaging the 4 Distinct Datasets...")
    os.makedirs("mal_ed_data/multi_targets", exist_ok=True)
    
    # Drop original target variables to prevent leaking
    z_cols = [c for c in df_visits.columns if 'z-score' in c.lower()]
    base_features = df_visits.drop(columns=z_cols + ["time_dt", "next_agedays"]).copy()
    
    # Dataset 1: BAZ Autoregressive
    ds1 = base_features.copy()
    # It keeps: target_prev, target_velocity
    # Target is: target (which is next BAZ)
    ds1.to_parquet("mal_ed_data/multi_targets/dataset_baz_ar.parquet")
    print(f" -> dataset_baz_ar.parquet ({len(ds1)} rows)")
    
    # Dataset 2: BAZ Delta
    ds2 = base_features.copy()
    ds2["target_delta"] = ds2["target"] - ds2["target_prev"]
    # CRITICAL: Drop absolute target_prev and target to force prediction purely on Delta!
    ds2 = ds2.drop(columns=["target", "target_prev"])
    ds2.to_parquet("mal_ed_data/multi_targets/dataset_baz_delta.parquet")
    print(f" -> dataset_baz_delta.parquet ({len(ds2)} rows)")
    
    # Dataset 3: Diarrhea Classification
    ds3 = base_features.copy()
    ds3["classification_target"] = ds3["target_diarrhea_bin"]
    ds3 = ds3.drop(columns=["target", "future_diarrhea_count", "future_illness_count", "target_diarrhea_bin"])
    ds3.to_parquet("mal_ed_data/multi_targets/dataset_diarrhea.parquet")
    print(f" -> dataset_diarrhea.parquet ({len(ds3)} rows)")
    
    # Dataset 4: Illness Burden Regression
    ds4 = base_features.copy()
    ds4["burden_target"] = ds4["future_illness_count"]
    ds4 = ds4.drop(columns=["target", "future_diarrhea_count", "future_illness_count", "target_diarrhea_bin"])
    ds4.to_parquet("mal_ed_data/multi_targets/dataset_illness_burden.parquet")
    print(f" -> dataset_illness_burden.parquet ({len(ds4)} rows)")
    
    print("\nMulti-Target Pipeline Complete!")

if __name__ == "__main__":
    build_pipeline()
