import pandas as pd
import numpy as np
import os

def binarize(series):
    series_str = series.astype(str).str.lower().str.strip()
    return series_str.isin(["yes", "1", "true"]).astype(int)

# --- TARGET DEFINITIONS ---
ALLOWED_FEATURES = {
    "baz_ar": [
        "pid", "agedays", "target",
        "target_prev", "target_velocity",
        "burden_illness_30d", "burden_diarrhea_30d", "burden_antibiotics_30d",
        "recovery_days_since_illness", "recovery_days_since_diarrhea",
        "Weight (kg)", "Height (cm)", "WAMI index"
    ],
    "delta_baz": [
        "pid", "agedays", "target_delta",
        "target_velocity",
        "burden_illness_30d", "burden_diarrhea_30d", "burden_antibiotics_30d",
        "recovery_days_since_illness", "recovery_days_since_diarrhea", "recovery_days_since_antibiotics",
        "Any illness, caregiver report", "Diarrhea, caregiver report", "Fever, caregiver report", "Use of antibiotics, caregiver report",
        "Weight (kg)", "Height (cm)", "WAMI index", "Maternal education (years)"
    ],
    "diarrhea_safe": [
        "pid", "agedays", "classification_target",
        "burden_illness_30d", "burden_antibiotics_30d",
        "recovery_days_since_illness", "recovery_days_since_antibiotics",
        "Any illness, caregiver report", "Fever, caregiver report", "Use of antibiotics, caregiver report",
        "WAMI index", "Sanitation score", "Drinking water score"
    ],
    "diarrhea_full": [
        "pid", "agedays", "classification_target",
        "burden_illness_30d", "burden_antibiotics_30d",
        "recovery_days_since_illness", "recovery_days_since_antibiotics",
        "Any illness, caregiver report", "Fever, caregiver report", "Use of antibiotics, caregiver report",
        "WAMI index", "Sanitation score", "Drinking water score",
        "burden_diarrhea_30d", "recovery_days_since_diarrhea", "Diarrhea, caregiver report"
    ],
    "illness_burden": [
        "pid", "agedays", "burden_target",
        "burden_illness_30d", "burden_diarrhea_30d", "burden_antibiotics_30d",
        "recovery_days_since_illness", "recovery_days_since_diarrhea", "recovery_days_since_antibiotics",
        "Any illness, caregiver report", "Diarrhea, caregiver report", "Fever, caregiver report", "Use of antibiotics, caregiver report",
        "WAMI index"
    ]
}

def enforce_schema(df, target_name):
    allowed = ALLOWED_FEATURES[target_name]
    df_clean = df[[c for c in allowed if c in df.columns]].copy()
    assert set(df_clean.columns) == set(allowed), f"Schema Error for {target_name}: Missing expected feature. Had {list(df_clean.columns)}"
    return df_clean

def build_v1_pipeline():
    print("Loading Original Clean Pipeline Dataset...")
    df = pd.read_parquet("mal_ed_data/mal_ed_final.parquet")
    df = df.sort_values(["pid", "agedays"]).reset_index(drop=True)
    df["time_dt"] = pd.to_timedelta(df["agedays"], unit="D")
    
    print("Engineering Burden and Recovery Matrices...")
    col_illness = "Any illness, caregiver report" if "Any illness, caregiver report" in df.columns else [c for c in df.columns if "any illness" in c.lower()][0]
    col_diar = "Diarrhea, caregiver report" if "Diarrhea, caregiver report" in df.columns else [c for c in df.columns if "diarrhea, caregiver report" in c.lower()][0]
    col_abx = "Use of antibiotics, caregiver report" if "Use of antibiotics, caregiver report" in df.columns else [c for c in df.columns if "antibiotic" in c.lower()][0]
    
    # Store standard names to match schema
    if col_illness != "Any illness, caregiver report": df["Any illness, caregiver report"] = df[col_illness]
    if col_diar != "Diarrhea, caregiver report": df["Diarrhea, caregiver report"] = df[col_diar]
    if col_abx != "Use of antibiotics, caregiver report": df["Use of antibiotics, caregiver report"] = df[col_abx]
    
    # Add Fever if missing in schema
    col_fever = "Fever, caregiver report" if "Fever, caregiver report" in df.columns else [c for c in df.columns if "fever" in c.lower()][0]
    if col_fever != "Fever, caregiver report": df["Fever, caregiver report"] = df[col_fever]
    
    df["ev_illness"] = binarize(df["Any illness, caregiver report"])
    df["ev_diarrhea"] = binarize(df["Diarrhea, caregiver report"])
    df["ev_antibiotics"] = binarize(df["Use of antibiotics, caregiver report"])
    
    df.set_index("time_dt", inplace=True)
    g = df.groupby("pid")
    
    df["burden_illness_30d"] = g["ev_illness"].rolling("30D", closed="left").sum().reset_index(level=0, drop=True)
    df["burden_diarrhea_30d"] = g["ev_diarrhea"].rolling("30D", closed="left").sum().reset_index(level=0, drop=True)
    df["burden_antibiotics_30d"] = g["ev_antibiotics"].rolling("30D", closed="left").sum().reset_index(level=0, drop=True)
    df.reset_index(inplace=True)
    df[["burden_illness_30d", "burden_diarrhea_30d", "burden_antibiotics_30d"]] = df[["burden_illness_30d", "burden_diarrhea_30d", "burden_antibiotics_30d"]].fillna(0)
    
    for ev_col, out_col in [("ev_illness", "recovery_days_since_illness"), 
                            ("ev_diarrhea", "recovery_days_since_diarrhea"), 
                            ("ev_antibiotics", "recovery_days_since_antibiotics")]:
        event_days = df["agedays"].where(df[ev_col] == 1)
        last_event_day = event_days.groupby(df["pid"]).ffill().groupby(df["pid"]).shift(1)
        df[out_col] = df["agedays"] - last_event_day
        df[out_col] = df[out_col].fillna(999)

    print("Isolating Clinical Visits and Anchoring Targets...")
    target_col = "BMI-for-age z-score"
    df_visits = df.dropna(subset=[target_col]).copy()
    df_visits = df_visits.sort_values(["pid", "agedays"])
    
    df_visits["target"] = df_visits.groupby("pid")[target_col].shift(-1)
    df_visits["target_prev"] = df_visits[target_col]
    df_visits["next_agedays"] = df_visits.groupby("pid")["agedays"].shift(-1)
    
    delta = df_visits.groupby("pid")[target_col].diff()
    time_diff = df_visits.groupby("pid")["agedays"].diff()
    df_visits["target_velocity"] = (delta / time_diff)
    df_visits["target_velocity"] = df_visits["target_velocity"].replace([np.inf, -np.inf], 0).fillna(0).clip(-5, 5)
    
    df_visits = df_visits.dropna(subset=["target", "next_agedays", "target_prev"])
    
    print("Extracting Next-Window Illness Horizons...")
    diar_between = []
    illness_between = []
    df_indexed = df.set_index(["pid", "agedays"])
    
    pids = df_visits["pid"].values
    t1 = df_visits["agedays"].values
    t2 = df_visits["next_agedays"].values
    
    for p, start_day, end_day in zip(pids, t1, t2):
        try:
            subset = df_indexed.loc[p]
            mask = (subset.index > start_day) & (subset.index <= end_day)
            slice_df = subset.loc[mask]
            diar_between.append(slice_df["ev_diarrhea"].sum())
            illness_between.append(slice_df["ev_illness"].sum())
        except KeyError:
            diar_between.append(0)
            illness_between.append(0)
            
    df_visits["classification_target"] = (np.array(diar_between) > 0).astype(int)
    df_visits["burden_target"] = illness_between
    
    df_visits["target_delta"] = df_visits["target"] - df_visits["target_prev"]

    # Fill NaNs specifically based on feature semantics
    def clean_features(export_df):
        for c in export_df.columns:
            if c in ["target", "target_delta", "classification_target", "burden_target", "pid"]:
                continue
            if export_df[c].dtype == "object":
                export_df[c] = export_df[c].fillna("Missing")
                export_df[c] = export_df[c].replace({"Unknown": "Missing", "unknown": "Missing", "": "Missing"})
            elif export_df[c].dtype.name == "category":
                if "Missing" not in export_df[c].cat.categories:
                    export_df[c] = export_df[c].cat.add_categories("Missing")
                export_df[c] = export_df[c].fillna("Missing")
                export_df[c] = export_df[c].replace({"Unknown": "Missing", "unknown": "Missing", "": "Missing"})
            else:
                if "score" in c.lower() or "index" in c.lower() or "education" in c.lower() or "Weight" in c or "Height" in c:
                    export_df[c] = export_df[c].fillna(export_df[c].median())
                else:
                    export_df[c] = export_df[c].fillna(0)
        return export_df

    print("Packaging the V1 Distinct Datasets...")
    os.makedirs("mal_ed_data/multi_targets/v1", exist_ok=True)
    
    ds_baz = enforce_schema(df_visits, "baz_ar")
    clean_features(ds_baz).to_parquet("mal_ed_data/multi_targets/v1/dataset_baz_ar.parquet")
    print(" -> dataset_baz_ar.parquet saved")
    
    ds_delta = enforce_schema(df_visits, "delta_baz")
    clean_features(ds_delta).to_parquet("mal_ed_data/multi_targets/v1/dataset_baz_delta.parquet")
    print(" -> dataset_baz_delta.parquet saved")
    
    ds_diar_safe = enforce_schema(df_visits, "diarrhea_safe")
    clean_features(ds_diar_safe).to_parquet("mal_ed_data/multi_targets/v1/dataset_diarrhea_v1_safe.parquet")
    print(" -> dataset_diarrhea_v1_safe.parquet saved")
    
    ds_diar_full = enforce_schema(df_visits, "diarrhea_full")
    clean_features(ds_diar_full).to_parquet("mal_ed_data/multi_targets/v1/dataset_diarrhea_v1_full.parquet")
    print(" -> dataset_diarrhea_v1_full.parquet saved")
    
    ds_burden = enforce_schema(df_visits, "illness_burden")
    clean_features(ds_burden).to_parquet("mal_ed_data/multi_targets/v1/dataset_illness_burden.parquet")
    print(" -> dataset_illness_burden.parquet saved")
    
    print("V1 Multi-Target Pipeline Complete!")

if __name__ == "__main__":
    build_v1_pipeline()
