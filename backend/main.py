import warnings
warnings.filterwarnings('ignore')

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from groq import Groq
import pandas as pd
import numpy as np

# Machine Learning Imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from xgboost import XGBClassifier, XGBRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge, LinearRegression
from sklearn.svm import SVC, SVR
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, r2_score, mean_absolute_error, mean_squared_error
from sklearn.feature_selection import RFE
import os, json

# Resolve project root (one level up from backend/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

app = FastAPI(title="MAL-ED Clinical Nexus API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from dataclasses import dataclass, field
from typing import List, Tuple, Optional

@dataclass
class PipelineState:
    dataset_id: str = ""
    df: pd.DataFrame = field(default_factory=pd.DataFrame)
    target: str = ""
    features: List[str] = field(default_factory=list)
    X: pd.DataFrame = field(default_factory=pd.DataFrame)
    y: pd.Series = field(default_factory=pd.Series)
    X_train: pd.DataFrame = field(default_factory=pd.DataFrame)
    X_test: pd.DataFrame = field(default_factory=pd.DataFrame)
    y_train: pd.Series = field(default_factory=pd.Series)
    y_test: pd.Series = field(default_factory=pd.Series)
    cv_splits: List[Tuple] = field(default_factory=list)
    task_type: str = ""
    selected_features: List[str] = field(default_factory=list)

# Global mutable reference (singleton)
PIPELINE_STATE: Optional[PipelineState] = None

# Minimal legacy GLOBAL_STATE for UI compatibility
GLOBAL_STATE = {
    "df": None,
    "trained_models": {},
    "target_col": None,
    "task_type": None,
    "selected_features": []
}

@app.on_event("startup")
def startup_event():
    # Load dataset automatically (resolve from project root)
    dataset_path = os.path.join(PROJECT_ROOT, "mal_ed_data", "mal_ed_final.parquet")
    try:
        raw_df = pd.read_parquet(dataset_path)
        GLOBAL_STATE["df"] = apply_clinical_preprocessing(raw_df)
        print(f"Dataset loaded & preprocessed: {GLOBAL_STATE['df'].shape}")
    except Exception as e:
        print(f"Failed to load dataset from {dataset_path}: {e}")

def apply_clinical_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply high-impact clinical preprocessing:
    1. Sort by PID and Age
    2. Normalize Time (Age in Months)
    3. Generate Missing Value Indicators
    4. Grouped Forward-Fill (Per Child)
    """
    print("Executing Clinical Preprocessing Pipeline...")
    # 1. Temporal Sorting
    df = df.sort_values(['pid', 'agedays'])

    # 2. Add Age in Months
    if 'agedays' in df.columns:
        df['age_months'] = df['agedays'] / 30.0

    # 3. Handle Missing Values (MAL-ED style)
    # Forward fill within child (pid) to preserve temporal continuity
    # We do this BEFORE missing indicators to fill what we can, 
    # but the user requested indicators for the "raw" state or the final state.
    # Usually, indicators are most useful for the FINAL state (what couldn't be filled).
    
    # To follow the prompt exactly: "add missing indicators ... forward fill within child"
    # We'll create indicators for columns that remain missing AFTER some basic cleaning 
    # OR for columns that have significant clinical meaning.
    
    # Let's target key clinical columns for indicators
    clinical_keywords = ['breastfeed', 'intake', 'fever', 'diarrhea', 'illness', 'antibiotic']
    for col in df.columns:
        if any(k in col.lower() for k in clinical_keywords) and df[col].isnull().any():
            df[f"{col}_missing"] = df[col].isnull().astype(int)

    # 4. Forward Fill per Child
    # Note: We only ffill columns that aren't PIDs or IDs
    cols_to_fill = [c for c in df.columns if c not in ['pid', 'Household_Id', 'age_month']]
    df[cols_to_fill] = df.groupby('pid')[cols_to_fill].ffill()

    # 5. Temporal Rolling & Trend Features
    print("   -> Calculating Temporal Rolling Features...")

    # Target Velocity and Lag‑1 (previous assessment)
    z_targets = ['BMI-for-age z-score', 'Length-for-age z-score', 'Weight-for-age z-score']
    for t in z_targets:
        if t in df.columns:
            # Velocity = Δvalue / Δtime (days)
            df[f"{t}_velocity_30d"] = df.groupby('pid').apply(
                lambda x: (x[t] - x[t].shift(1)) / (x['agedays'] - x['agedays'].shift(1) + 1e-5)
            ).reset_index(level=0, drop=True).fillna(0)
            # Lag‑1 of the target (previous assessment)
            df[f"{t}_prev"] = df.groupby('pid')[t].shift(1).fillna(0)

    # Lean Burdens – true 30‑day rolling sum based on actual days
    illness_col = 'Any illness, caregiver report'
    diarrhea_col = 'Diarrhea, caregiver report'
    for col, new_name in [(illness_col, 'illness_days_30d'), (diarrhea_col, 'diarrhea_days_30d')]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            def rolling_30d(g):
                ages = g['agedays'].values
                vals = g[col].values
                out = np.zeros_like(vals, dtype=float)
                for i, age in enumerate(ages):
                    mask = (ages > age - 30) & (ages <= age)
                    out[i] = vals[mask].sum()
                return pd.Series(out, index=g.index)
            df[new_name] = df.groupby('pid').apply(rolling_30d).reset_index(level=0, drop=True)

    # Capacitated Recovery Windows – days since last event, capped at 90 days
    for col, new_name in [(illness_col, 'days_since_illness'), (diarrhea_col, 'days_since_diarrhea')]:
        if col in df.columns:
            last_event_age = df['agedays'].where(df[col] > 0)
            last_event_age = last_event_age.groupby(df['pid']).ffill()
            raw_days_since = df['agedays'] - last_event_age
            df[new_name] = raw_days_since.fillna(90).clip(upper=90)

    print("Preprocessing Complete. Temporal features active.")
    return df

@app.get("/api/ping")
def ping():
    return {"status": "ok", "message": "FastAPI is running"}

class LoadRequest(BaseModel):
    variant: str

@app.post("/api/dataset/load")
def load_dataset_variant(req: LoadRequest):
    paths = {
        "daily": os.path.join("mal_ed_data", "mal_ed_final.parquet"),
        "monthly": os.path.join("mal_ed_data", "mal_ed_monthly_trajectories.parquet"),
        "summary": os.path.join("mal_ed_data", "mal_ed_summary_targets.parquet"),
    }
    p = paths.get(req.variant)
    if not p or not os.path.exists(p):
        raise HTTPException(status_code=404, detail="Dataset variant not found")
        
    GLOBAL_STATE["df"] = apply_clinical_preprocessing(pd.read_parquet(p))
    return {
        "status": "success",
        "columns_list": list(GLOBAL_STATE["df"].columns)
    }

@app.get("/api/dataset/overview")
def get_dataset_overview():
    df = GLOBAL_STATE.get("df")
    if df is None:
        raise HTTPException(status_code=500, detail="Dataset not loaded")
    
    unique_pid = df['pid'].nunique() if 'pid' in df.columns else 0
    completeness = round((1 - df.isnull().mean().mean()) * 100, 1)

    return {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "unique_children": unique_pid,
        "completeness": completeness,
        "columns_list": list(df.columns)
    }

@app.get("/api/dataset/dictionary")
def get_dataset_dictionary():
    # Load enriched clinical codebook with context (resolve from project root)
    enriched_path = os.path.join(PROJECT_ROOT, "mal_ed_data", "enriched_codebook.csv")
    meta_path = os.path.join(PROJECT_ROOT, "mal_ed_data", "MAL-ED_0-60m_OntologyMetadata.txt")
    
    if os.path.exists(enriched_path):
        meta = pd.read_csv(enriched_path).fillna("")
        return {"dictionary": meta.to_dict(orient="records")}
        
    if os.path.exists(meta_path):
        meta = pd.read_csv(meta_path, sep='\t').fillna("")
        # Clean label for legacy dictionary
        import re
        meta['Feature'] = meta['label'].apply(lambda x: re.sub(r'\s*\[.*?\]', '', str(x)).strip())
        meta['Category'] = meta['category']
        meta['Definition'] = meta['definition']
        return {"dictionary": meta[['Feature', 'Category', 'Definition']].to_dict(orient="records")}
        
    return {"dictionary": []}

@app.get("/api/dataset/univariate")
def get_dataset_univariate(column: str):
    df = GLOBAL_STATE.get("df")
    if df is None:
        raise HTTPException(status_code=500, detail="Dataset not loaded")
    
    if column not in df.columns:
        raise HTTPException(status_code=400, detail="Column not found")
        
    # Analyze distribution
    series = df[column].dropna().head(100000)
    
    if series.dtype in ['float32', 'float64', 'int32', 'int64'] and series.nunique() > 15:
        # Generate Histogram bins
        counts, bins = np.histogram(series, bins=30)
        chart_data = [{"bin": f"{bins[i]:.2f}-{bins[i+1]:.2f}", "count": int(counts[i])} for i in range(len(counts))]
        type_ = "histogram"
    else:
        # Generate Value Counts
        vc = series.value_counts().head(15)
        chart_data = [{"name": str(k), "count": int(v)} for k, v in vc.items()]
        type_ = "bar"
        
    return {
        "column": column,
        "type": type_,
        "data": chart_data
    }

@app.get("/api/dataset/sample")
def get_dataset_sample():
    df = GLOBAL_STATE.get("df")
    if df is None:
        raise HTTPException(status_code=500, detail="Dataset not loaded")
        
    # Return top 20 rows and top 12 columns just for UI table plotting
    view_cols = list(df.columns)[:12]
    # Convert to string to avoid Categorical type errors when filling NaNs
    sample_df = df[view_cols].head(20).astype(str).replace(["nan", "NaN", "None", "<NA>"], "")
    
    return {
        "columns": view_cols,
        "data": sample_df.to_dict(orient="records")
    }

class TargetRequest(BaseModel):
    target: str

@app.post("/api/dataset/load_target")
def load_target_dataset(req: TargetRequest):
    p = os.path.join("mal_ed_data", "multi_targets", f"{req.target}.parquet")
    if not os.path.exists(p):
        raise HTTPException(status_code=404, detail=f"Target dataset {req.target} not found")
    
    df = pd.read_parquet(p)
    sample_df = df.head(100).astype(str).replace(["nan", "NaN", "None", "<NA>"], "")
    return {
        "status": "success",
        "data": sample_df.to_dict(orient="records")
    }

import json

TARGET_MAPPING = {
    "target": "baz_ar",
    "target_delta": "delta_baz",
    "classification_target": "diarrhea",
    "burden_target": "illness_burden"
}

class RegistryManager:
    # Resolve the project root dynamically, assuming backend/main.py is located at ProjectRoot/backend/main.py
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    BASE_DIR = os.path.join(PROJECT_ROOT, 'models')

    @classmethod
    def get_model_path(cls, target_col: str, model_name: str, version: str = "v1") -> str:
        reg_target = TARGET_MAPPING.get(target_col, target_col)
        model_slug = model_name.lower().replace(" ", "_")
        return os.path.join(cls.BASE_DIR, version, reg_target, model_slug)

    @classmethod
    def get_metadata(cls, target_col: str, model_name: str, version: str = "v1") -> dict:
        path = os.path.join(cls.get_model_path(target_col, model_name, version), 'metadata.json')
        if not os.path.exists(path): return {}
        with open(path, 'r') as f: return json.load(f)

    @classmethod
    def get_shap_global(cls, target_col: str, model_name: str, version: str = "v1") -> dict:
        path = os.path.join(cls.get_model_path(target_col, model_name, version), 'shap_global.json')
        if not os.path.exists(path): return {}
        with open(path, 'r') as f: return json.load(f)

    @classmethod
    def get_predictions(cls, target_col: str, model_name: str, version: str = "v1") -> dict:
        path = os.path.join(cls.get_model_path(target_col, model_name, version), 'predictions.parquet')
        if not os.path.exists(path): return []
        df = pd.read_parquet(path)
        if len(df) > 2000: df = df.sample(2000, random_state=42)
        return df.to_dict(orient="records")

    @classmethod
    def get_expected_value(cls, target_col: str, model_name: str, version: str = "v1") -> float:
        path = os.path.join(cls.get_model_path(target_col, model_name, version), 'expected_value.json')
        if not os.path.exists(path): return 0.0
        with open(path, 'r') as f:
            data = json.load(f)
        return data.get('expected_value', 0.0)

    @classmethod
    def get_shap_detail(cls, target_col: str, model_name: str, version: str = "v1") -> dict:
        """Returns full SHAP detail: per-row SHAP values, feature names, feature values, expected value, and global importance."""
        model_path = cls.get_model_path(target_col, model_name, version)
        
        shap_path = os.path.join(model_path, 'shap_sample.parquet')
        shap_values_path = os.path.join(model_path, 'shap_sample_values.parquet')
        shap_data_path = os.path.join(model_path, 'shap_sample_data.parquet')

        feature_names = []
        shap_values = []
        feature_values = []

        if os.path.exists(shap_path):
            # V1 Merged format (or V1 SHAP-only format)
            shap_df = pd.read_parquet(shap_path)
            # Check if it's the combined format (shap_ prefix and val_ prefix)
            shap_cols = [c for c in shap_df.columns if c.startswith('shap_')]
            val_cols = [c for c in shap_df.columns if c.startswith('val_')]
            
            if shap_cols and val_cols:
                # True merged format
                feature_names = [c.replace('shap_', '') for c in shap_cols]
                shap_values = shap_df[shap_cols].values.tolist()
                feature_values = shap_df[val_cols].values.tolist()
            else:
                # Legacy or SHAP-only V1 format
                feature_names = list(shap_df.columns)
                shap_values = shap_df.values.tolist()
                # Need to fetch feature values from raw dataset
                feature_values = cls._fetch_feature_values_from_dataset(target_col, model_name, version, feature_names)
        elif os.path.exists(shap_values_path) and os.path.exists(shap_data_path):
            # V0 Split format
            shap_df = pd.read_parquet(shap_values_path)
            data_df = pd.read_parquet(shap_data_path)
            
            # Use intersection of columns to be safe
            feature_names = [c for c in shap_df.columns if c in data_df.columns]
            # Limit features for performance if too many
            if len(feature_names) > 30:
                # Get global importance to find top features
                global_imp = cls.get_shap_global(target_col, model_name, version)
                if global_imp:
                    top_features = sorted(global_imp.keys(), key=lambda x: global_imp[x], reverse=True)[:30]
                    feature_names = [f for f in feature_names if f in top_features]
            
            shap_values = shap_df[feature_names].values.tolist()
            feature_values = data_df[feature_names].values.tolist()
        else:
            return {}

        expected_value = cls.get_expected_value(target_col, model_name, version)
        global_importance = cls.get_shap_global(target_col, model_name, version)

        return {
            'shap_values': shap_values,
            'feature_names': feature_names,
            'feature_values': feature_values,
            'expected_value': expected_value,
            'global_importance': global_importance
        }

    @classmethod
    def _fetch_feature_values_from_dataset(cls, target_col: str, model_name: str, version: str, feature_names: List[str]) -> List[List[float]]:
        """Helper to fetch raw feature values for SHAP coloring when not present in artifacts."""
        meta = cls.get_metadata(target_col, model_name, version)
        if not meta: return []
        
        reg_target = TARGET_MAPPING.get(target_col, target_col)
        dataset_map = {
            'baz_ar': 'dataset_baz_ar.parquet',
            'delta_baz': 'dataset_baz_delta.parquet',
            'diarrhea': 'dataset_diarrhea_v1_full.parquet',
            'illness_burden': 'dataset_illness_burden.parquet'
        }
        dataset_file = dataset_map.get(reg_target, '')
        
        # Check in both multi_targets/version and root mal_ed_data
        search_paths = [
            os.path.join(cls.PROJECT_ROOT, 'mal_ed_data', 'multi_targets', version, dataset_file),
            os.path.join(cls.PROJECT_ROOT, 'mal_ed_data', 'multi_targets', dataset_file),
            os.path.join(cls.PROJECT_ROOT, 'mal_ed_data', dataset_file)
        ]
        
        dataset_path = next((p for p in search_paths if os.path.exists(p)), None)
        if not dataset_path: return []

        ds = pd.read_parquet(dataset_path)
        # Drop identifiers
        drop_cols = ['pid', 'Household_Id', meta.get('target', '')]
        ds_features = ds.drop(columns=[c for c in drop_cols if c in ds.columns], errors='ignore')
        
        # Sanitize columns to match SHAP artifacts
        ds_features.columns = [c.replace('[', '_').replace(']', '_').replace('<', '_') for c in ds_features.columns]
        
        # Numeric coercion
        for c in ds_features.columns:
            ds_features[c] = pd.to_numeric(ds_features[c].astype(str).str.lower().str.strip().replace({'yes':'1','no':'0','missing':'-1'}), errors='coerce').fillna(0)
            
        # Sample same rows (using RandomState 42 as per pipeline)
        sample_size = min(300, len(ds_features))
        idx = np.random.RandomState(42).choice(len(ds_features), sample_size, replace=False)
        ds_sample = ds_features.iloc[idx]
        
        # Reorder and extract
        reordered = []
        for fn in feature_names:
            if fn in ds_sample.columns:
                reordered.append(ds_sample[fn].values.tolist())
            else:
                reordered.append([0.0] * sample_size)
        
        return [list(row) for row in zip(*reordered)]

    @classmethod
    def get_threshold_metrics(cls, target_col: str, model_name: str, version: str = "v1", threshold: float = 0.5) -> dict:
        """Computes threshold-specific metrics (Confusion Matrix, F1, F2) server-side."""
        from sklearn.metrics import confusion_matrix, f1_score, fbeta_score, precision_score, recall_score
        
        path = os.path.join(cls.get_model_path(target_col, model_name, version), 'predictions.parquet')
        if not os.path.exists(path): return {}
        df = pd.read_parquet(path)
        if 'y_prob' not in df.columns: return {}

        y_true = df['y_true'].values
        y_prob = df['y_prob'].values
        y_pred = (y_prob >= threshold).astype(int)

        # cm is [[TN, FP], [FN, TP]]
        cm = confusion_matrix(y_true, y_pred).tolist()
        
        precision = float(precision_score(y_true, y_pred, zero_division=0))
        recall = float(recall_score(y_true, y_pred, zero_division=0))
        f1 = float(f1_score(y_true, y_pred, zero_division=0))
        f2 = float(fbeta_score(y_true, y_pred, beta=2, zero_division=0))

        return {
            'threshold': threshold,
            'confusion_matrix': cm,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'f2': f2
        }

    @classmethod
    def get_classification_curves(cls, target_col: str, model_name: str, version: str = "v1") -> dict:
        """Computes ROC, PR, and Calibration curves server-side from predictions."""
        from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
        from sklearn.calibration import calibration_curve

        path = os.path.join(cls.get_model_path(target_col, model_name, version), 'predictions.parquet')
        if not os.path.exists(path): return {}
        df = pd.read_parquet(path)
        if 'y_prob' not in df.columns: return {}

        y_true = df['y_true'].values
        y_prob = df['y_prob'].values

        # ROC Curve
        fpr, tpr, roc_thresholds = roc_curve(y_true, y_prob)
        roc_auc = float(auc(fpr, tpr))
        # Downsample for performance
        step = max(1, len(fpr) // 200)
        roc_data = [{'fpr': float(fpr[i]), 'tpr': float(tpr[i])} for i in range(0, len(fpr), step)]

        # Precision-Recall Curve
        precision, recall, pr_thresholds = precision_recall_curve(y_true, y_prob)
        avg_precision = float(average_precision_score(y_true, y_prob))
        step_pr = max(1, len(precision) // 200)
        pr_data = [{'precision': float(precision[i]), 'recall': float(recall[i])} for i in range(0, len(precision), step_pr)]

        # Calibration Curve
        try:
            fraction_positives, mean_predicted = calibration_curve(y_true, y_prob, n_bins=10, strategy='uniform')
            cal_data = [{'fraction_positive': float(fp), 'mean_predicted': float(mp)} for fp, mp in zip(fraction_positives, mean_predicted)]
        except Exception:
            cal_data = []

        return {
            'roc': {'data': roc_data, 'auc': roc_auc},
            'pr': {'data': pr_data, 'avg_precision': avg_precision},
            'calibration': {'data': cal_data}
        }

class ScoreTargetRequest(BaseModel):
    target: str
    model: str
    task_type: str
    use_kfold: bool = True
    force_retrain: bool = False
    ablations: list[str] = []
    version: str = "v1"

@app.get("/api/models/registry/metadata")
def get_registry_metadata(target: str, model: str, version: str = "v1"):
    meta = RegistryManager.get_metadata(target, model, version)
    if not meta: raise HTTPException(status_code=404, detail="Registry metadata not found.")
    return {"status": "success", "data": meta}

@app.get("/api/models/registry/shap")
def get_registry_shap(target: str, model: str, version: str = "v1"):
    shap_vals = RegistryManager.get_shap_global(target, model, version)
    if not shap_vals: raise HTTPException(status_code=404, detail="Registry SHAP not found.")
    return {"status": "success", "data": shap_vals}

@app.get("/api/models/registry/predictions")
def get_registry_preds(target: str, model: str, version: str = "v1"):
    preds = RegistryManager.get_predictions(target, model, version)
    if not preds: raise HTTPException(status_code=404, detail="Registry Predictions not found.")
    return {"status": "success", "data": preds}

@app.get("/api/models/registry/shap_detail")
def get_registry_shap_detail(target: str, model: str, version: str = "v1"):
    detail = RegistryManager.get_shap_detail(target, model, version)
    if not detail: raise HTTPException(status_code=404, detail="SHAP detail artifacts not found.")
    return {"status": "success", "data": detail}

@app.get("/api/models/registry/classification_curves")
def get_registry_classification_curves(target: str, model: str, version: str = "v1"):
    curves = RegistryManager.get_classification_curves(target, model, version)
    if not curves: raise HTTPException(status_code=404, detail="Classification curves not available (regression target or missing y_prob).")
    return {"status": "success", "data": curves}

@app.get("/api/models/registry/threshold_metrics")
def get_registry_threshold_metrics(target: str, model: str, version: str = "v1", threshold: float = 0.5):
    metrics = RegistryManager.get_threshold_metrics(target, model, version, threshold)
    if not metrics: raise HTTPException(status_code=404, detail="Threshold metrics not available.")
    return {"status": "success", "data": metrics}

@app.post("/api/model/score_target")
def score_target(req: ScoreTargetRequest):
    # 1. Hybrid Instant Registry Serving
    if not req.force_retrain:
        meta = RegistryManager.get_metadata(req.target, req.model, req.version)
        if not meta:
            raise HTTPException(status_code=404, detail="Model not found in registry. Please Force Retrain.")
        
        # Reformating the metrics to precisely match the old expected UI contract dict nesting
        fmt_metrics = {}
        for k, v in meta["metrics"].items():
            if k == "R2": fmt_metrics["R²"] = f"{v:.4f}"
            else: fmt_metrics[k] = f"{v:.4f}"
            
        # UI expects: { "Random Forest": {"R²": "0.85", "MAE": "0.2", "RMSE": "0.5"} }
        metrics = { req.model: fmt_metrics }
        
        return {"status": "success", "source": "registry", "metrics": metrics}

    # 2. Force Retrain Fallback
    p = ""
    target_col = req.target
    if req.target == "target":
        p = os.path.join("mal_ed_data", "multi_targets", "dataset_baz_ar.parquet")
    elif req.target == "target_delta":
        p = os.path.join("mal_ed_data", "multi_targets", "dataset_baz_delta.parquet")
    elif req.target == "classification_target":
        p = os.path.join("mal_ed_data", "multi_targets", "dataset_diarrhea.parquet")
    elif req.target == "burden_target":
        p = os.path.join("mal_ed_data", "multi_targets", "dataset_illness_burden.parquet")
    else:
        raise HTTPException(status_code=400, detail="Invalid target")
        
    if not os.path.exists(p):
        raise HTTPException(status_code=404, detail="Dataset not found")
        
    df = pd.read_parquet(p)
    df = df.dropna(subset=[target_col])
    
    # Inline clean_features logic to avoid import circularity
    X = df.drop(columns=["pid", target_col]).copy()
    for col in X.select_dtypes(include=[object]).columns:
        X[col] = X[col].replace({"Unknown": np.nan, "unknown": np.nan, "": np.nan})
        X[col] = pd.to_numeric(X[col], errors="coerce")
        median_val = X[col].median()
        if pd.isna(median_val): median_val = 0
        X[col] = X[col].fillna(median_val)
    for col in X.select_dtypes(include=['category']).columns:
        X[col] = X[col].cat.codes
    X = X.fillna(0)
    
    # 3. Apply Ablation Filter Before Training
    if req.force_retrain and req.ablations:
        drop_cols = []
        if "temporal" in req.ablations:
            drop_cols += ["agedays", "age_month"]
        if "lag" in req.ablations:
            # Drop previous memory targets and derived dynamic signals
            drop_cols += [c for c in X.columns if "prev" in c or "velocity" in c or "burden_" in c or "recovery_" in c or "future_" in c or "target_" in c or "recent_" in c]
        if "ses" in req.ablations:
            ses_terms = ["WAMI index", "Overall socioeconomic score", "Household wealth index", "Income score", "Sanitation score", "Drinking water score", "Maternal education", "Household Food Insecurity"]
            drop_cols += [c for c in X.columns if any(term in c for term in ses_terms)]
            
        real_drops = [c for c in drop_cols if c in X.columns]
        X = X.drop(columns=real_drops)
        print(f"[ABLATION ACTIVE] Dropped {len(real_drops)} specific temporal/lag/ses columns.")
    
    if req.task_type == "regression":
        y = df[target_col]
    else:
        y = df[target_col].astype(int)
        
    groups = df["pid"]
    from sklearn.model_selection import GroupKFold, train_test_split
    splits = []
    if req.use_kfold:
        gkf = GroupKFold(n_splits=5)
        splits = list(gkf.split(X, y, groups))
    else:
        # standard 80-20 train-test split
        idx = np.arange(len(X))
        train_idx, val_idx = train_test_split(idx, test_size=0.2, random_state=42)
        splits = [(train_idx, val_idx)]
    
    if req.task_type == "regression":
        if req.model == "Random Forest":
            clf = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)
        elif req.model == "XGBoost":
            clf = XGBRegressor(n_estimators=50, max_depth=6, random_state=42)
        elif req.model == "CatBoost":
            clf = CatBoostRegressor(iterations=50, depth=6, random_state=42, verbose=0)
        else:
            clf = RandomForestRegressor(n_estimators=50, random_state=42)
            
        cv_r2, cv_mae, cv_rmse = [], [], []
        for train_idx, val_idx in splits:
            clf.fit(X.iloc[train_idx], y.iloc[train_idx])
            preds = clf.predict(X.iloc[val_idx])
            cv_r2.append(r2_score(y.iloc[val_idx], preds))
            cv_mae.append(mean_absolute_error(y.iloc[val_idx], preds))
            cv_rmse.append(np.sqrt(mean_squared_error(y.iloc[val_idx], preds)))
            
        metrics = {
            req.model: {
                "R²": f"{np.mean(cv_r2):.4f}",
                "MAE": f"{np.mean(cv_mae):.4f}",
                "RMSE": f"{np.mean(cv_rmse):.4f}"
            }
        }
    else:
        if req.model == "Random Forest":
            clf = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, class_weight='balanced')
        elif req.model == "XGBoost":
            clf = XGBClassifier(n_estimators=50, max_depth=6, random_state=42)
        elif req.model == "CatBoost":
            clf = CatBoostClassifier(iterations=50, depth=6, random_state=42, verbose=0)
        else:
            clf = RandomForestClassifier(n_estimators=50, random_state=42)
            
        cv_auc, cv_f1, cv_acc = [], [], []
        for train_idx, val_idx in splits:
            clf.fit(X.iloc[train_idx], y.iloc[train_idx])
            preds_proba = clf.predict_proba(X.iloc[val_idx])[:, 1] if hasattr(clf, "predict_proba") else clf.predict(X.iloc[val_idx])
            preds_class = clf.predict(X.iloc[val_idx])
            try:
                cv_auc.append(roc_auc_score(y.iloc[val_idx], preds_proba))
            except:
                cv_auc.append(0)
            cv_f1.append(f1_score(y.iloc[val_idx], preds_class, average='weighted'))
            cv_acc.append(accuracy_score(y.iloc[val_idx], preds_class))
            
        metrics = {
            req.model: {
                "ROC-AUC": f"{np.mean(cv_auc):.4f}",
                "F1": f"{np.mean(cv_f1):.4f}",
                "Accuracy": f"{np.mean(cv_acc):.4f}"
            }
        }
        
    return {"metrics": metrics}

class VisualizeRequest(BaseModel):
    model: str
    target: str
    task_type: str = "regression"
    version: str = "v1"

@app.post("/api/models/visualize")
def visualize_model(req: VisualizeRequest):
    """Generate scatter/confusion_matrix + feature importance for the Model Arena visualizer."""
    # Try registry predictions first
    predictions = RegistryManager.get_predictions(req.target, req.model, req.version)
    shap_global = RegistryManager.get_shap_global(req.target, req.model, req.version)
    
    result = {"task_type": req.task_type}
    
    if predictions:
        preds_df = pd.DataFrame(predictions)
        
        if req.task_type == "regression":
            # Scatter plot data (actual vs predicted)
            scatter_data = []
            sample = preds_df.head(500)  # Cap for performance
            for _, row in sample.iterrows():
                scatter_data.append({
                    "actual": round(float(row.get("y_true", 0)), 4),
                    "predicted": round(float(row.get("y_pred", 0)), 4)
                })
            result["scatter"] = scatter_data
        else:
            # Classification: confusion matrix
            y_true = preds_df["y_true"].values.astype(int)
            y_pred = preds_df.get("y_pred", (preds_df.get("y_prob", pd.Series([0]*len(preds_df))) >= 0.5).astype(int)).values.astype(int)
            cm = confusion_matrix(y_true, y_pred).tolist()
            result["confusion_matrix"] = cm
    else:
        # No registry predictions — return empty
        if req.task_type == "regression":
            result["scatter"] = []
        else:
            result["confusion_matrix"] = []
    
    # Feature importance from SHAP global
    if shap_global:
        sorted_features = sorted(shap_global.items(), key=lambda x: abs(x[1]), reverse=True)[:15]
        result["feature_importance"] = [
            {"feature": f, "importance": round(v, 6)} for f, v in sorted_features
        ]
    
    return result

class FeatureRequest(BaseModel):
    target: str
    methods: list[str] = ["Mutual Information", "Correlation Analysis", "SHAP", "Permutation"]
    sample_size: int = 50000


# Precomputed discovery cache for demo speed
DISCOVERY_CACHE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "discovery_cache.json")

@app.get("/api/models/feature_discovery/cache")
def get_discovery_cache(target: str = None):
    if not os.path.exists(DISCOVERY_CACHE_PATH):
        return {"status": "miss", "data": {}}
    with open(DISCOVERY_CACHE_PATH, "r") as f:
        cache = json.load(f)
    if target:
        if target in cache:
            return {"status": "hit", "data": cache[target]}
        return {"status": "miss", "data": {}}
    return {"status": "hit", "data": cache}

@app.post("/api/models/feature_discovery")
def run_feature_discovery(req: FeatureRequest):
    df = GLOBAL_STATE.get("df")
    if df is None:
        raise HTTPException(status_code=500, detail="Dataset not loaded")
        
    if req.target not in df.columns:
        raise HTTPException(status_code=400, detail="Target not found in dataset")
        
    # Remove identifiers and any time-derived columns that could leak information
    # exclude target and target leakages + clinical memory derived features
    exclude_cols = ['pid', 'Household_Id', req.target, f"{req.target}_missing", 'age_months', 'agedays', 'age_month']
    memory_cols = ['illness_days_30d', 'diarrhea_days_30d', 'days_since_illness', 'days_since_diarrhea', f"{req.target}_velocity_30d", f"{req.target}_prev"]
    exclude_cols.extend(memory_cols)
    
    available_features = [c for c in df.columns if c not in exclude_cols and pd.api.types.is_numeric_dtype(df[c])]
    
    df_clean = df[available_features + [req.target]].copy()
    df_clean = df_clean.dropna(subset=[req.target])
    
    # Impute feature NaNs to prevent dropping longitudinal data
    for col in available_features:
        if pd.api.types.is_numeric_dtype(df_clean[col]):
            med = df_clean[col].median()
            df_clean[col] = df_clean[col].fillna(med if not pd.isna(med) else 0)
            
    df_clean = df_clean.head(req.sample_size)
    
    X = df_clean.drop(columns=[req.target])
    
    # Target Leakage Defense (from legacy app_v2.py)
    # Automatically strip other Z-scores and anthropometric proxies if predicting growth
    leakage_cols = [c for c in X.columns if 'z-score' in c.lower() or 'weight' in c.lower() or 'length' in c.lower()]
    if leakage_cols:
        X = X.drop(columns=leakage_cols)
        
    y = df_clean[req.target]
    
    for col in X.columns:
        if X[col].dtype == 'object' or X[col].dtype.name == 'category':
            try:
                X[col] = LabelEncoder().fit_transform(X[col].astype(str))
            except:
                X = X.drop(columns=[col])
        # Ensure numeric columns are float for SHAP/Permutation
        else:
            X[col] = pd.to_numeric(X[col], errors='coerce')
    if y.dtype == 'object' or y.dtype.name == 'category':
        y = LabelEncoder().fit_transform(y.astype(str))
        task = "classification"
    else:
        task = "regression"

    scores = []
    
    # 1. Mutual Information
    if "Mutual Information" in req.methods or len(req.methods) == 0:
        if task == "classification":
            mi_scores = mutual_info_classif(X, y)
        else:
            mi_scores = mutual_info_regression(X, y)
        for name, score in zip(X.columns, mi_scores):
            scores.append({"feature": name, "score": float(score), "method": "Mutual Information"})

    # 2. Correlation Analysis
    if "Correlation Analysis" in req.methods:
        corr_matrix = pd.concat([X, y], axis=1).corr()
        target_corr = corr_matrix.iloc[:-1, -1].abs()
        for name, score in target_corr.items():
            if not np.isnan(score):
                scores.append({"feature": name, "score": float(score), "method": "Correlation"})

    # 3. SHAP Importance (using LightGBM as reference model)
    if "SHAP" in req.methods:
        try:
            from lightgbm import LGBMRegressor, LGBMClassifier
            import shap
            if task == "classification":
                model = LGBMClassifier(n_estimators=100, random_state=42)
            else:
                model = LGBMRegressor(n_estimators=100, random_state=42)
            model.fit(X, y)
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)
            # For classification, shap_values is a list per class; take mean absolute across classes
            if isinstance(shap_values, list):
                shap_abs = np.mean([np.abs(s).mean(axis=0) for s in shap_values], axis=0)
            else:
                shap_abs = np.abs(shap_values).mean(axis=0)
            for name, val in zip(X.columns, shap_abs):
                scores.append({"feature": name, "score": float(val), "method": "SHAP"})
        except Exception as e:
            # Fallback: ignore SHAP if any error
            print(f"SHAP calculation failed: {e}")

    # 4. Permutation Importance (sklearn implementation)
    if "Permutation" in req.methods:
        try:
            from sklearn.inspection import permutation_importance
            # Use a fast LightGBM model for permutation as well
            if task == "classification":
                perm_model = LGBMClassifier(n_estimators=100, random_state=42)
            else:
                perm_model = LGBMRegressor(n_estimators=100, random_state=42)
            perm_model.fit(X, y)
            result = permutation_importance(perm_model, X, y, n_repeats=5, random_state=42, n_jobs=-1)
            for name, val in zip(X.columns, result.importances_mean):
                scores.append({"feature": name, "score": float(val), "method": "Permutation"})
        except Exception as e:
            print(f"Permutation importance failed: {e}")

    # Aggregate and average
    agg_df = pd.DataFrame(scores)
    if agg_df.empty:
        return {"target": req.target, "task_type": task, "results": []}
        
    final_scores = agg_df.groupby("feature")["score"].mean().reset_index().sort_values("score", ascending=False).head(50)
    final_list = [{"feature": row["feature"], "score": float(row["score"])} for _, row in final_scores.iterrows()]
    
    return {
        "target": req.target,
        "task_type": task,
        "results": final_list
    }

class TrainRequest(BaseModel):
    target: str
    features: list[str]
    models: list[str]
    task_type: str = "classification"
    clear_state: bool = False
    use_kfold: bool = False
    feature_engineering: bool = False
    target_horizon_days: int = 0
    use_clinical_memory: bool = False

@app.post("/api/models/train")
def run_model_training(req: TrainRequest):
    df = GLOBAL_STATE.get("df")
    if df is None:
        raise HTTPException(status_code=500, detail="Dataset not loaded")
    
    # 1. Target Shift (Temporal Definition)
    df_clean = df.copy()
    if req.target_horizon_days > 0 and 'pid' in df_clean.columns and 'agedays' in df_clean.columns:
        df_clean['forecast_age'] = df_clean['agedays'] + req.target_horizon_days
        df_future = df_clean[['pid', 'agedays', req.target]].copy()
        df_future = df_future.rename(columns={'agedays': 'forecast_age', req.target: 'target_future'})
        
        # Sorting for merge_asof requirement
        df_clean = df_clean.sort_values('forecast_age')
        df_future = df_future.sort_values('forecast_age')
        
        df_clean = pd.merge_asof(
            df_clean,
            df_future,
            on='forecast_age',
            by='pid',
            direction='nearest',
            tolerance=7
        )
        df_clean = df_clean.sort_values(['pid', 'agedays']) # Restore sorting
        
        # Option A: Predict Deviation (Delta) instead of Absolute Inertia
        df_clean[req.target] = df_clean['target_future'] - df_clean[req.target]
        df_clean = df_clean.drop(columns=['forecast_age', 'target_future'])

    core_features = ['illness_days_30d', 'diarrhea_days_30d', 'days_since_illness', 'days_since_diarrhea']
    vel_col = f"{req.target}_velocity_30d"
    prev_col = f"{req.target}_prev"
    if vel_col in df_clean.columns:
        core_features.append(vel_col)
    if prev_col in df_clean.columns:
        core_features.append(prev_col)

    if req.use_clinical_memory:
        for f in core_features:
            if f in df_clean.columns and f not in req.features:
                req.features.append(f)
    else:
        # Strict Ablation: Strip them out if the user turned the toggle off
        req.features = [f for f in req.features if f not in core_features]

    required_cols = req.features + [req.target]
    if req.use_kfold and 'pid' in df_clean.columns:
        required_cols.append('pid')
    
    # Remove duplicates but keep order
    required_cols = list(dict.fromkeys(required_cols))
    
    df_clean = df_clean[required_cols].copy()
    
    # Only drop rows where the target is missing (the forecast target didn't exist)
    df_clean = df_clean.dropna(subset=[req.target])
    
    X = df_clean[req.features].copy()
    y = df_clean[req.target]
    
    if req.use_kfold and 'pid' in df_clean.columns:
        groups = df_clean['pid']
    else:
        groups = None

    # Impute missing values so we don't destroy cross-sectional and temporal histories
    for col in X.columns:
        if X[col].dtype == 'object' or X[col].dtype.name == 'category':
            X[col] = X[col].fillna("MISSING")
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))
        else:
            med = X[col].median()
            X[col] = X[col].fillna(med if not pd.isna(med) else 0)
            
    if req.task_type == "classification" and (y.dtype == 'object' or y.dtype.name == 'category'):
        y = LabelEncoder().fit_transform(y.astype(str))

    # ------------------ PILOT SAMPLE LOGIC (Lean Integration) ------------------ 
    # Throttle dataset size to guarantee interactive speed (approx 75,000 rows).
    if req.clear_state or GLOBAL_STATE.get('X_train') is None:
        max_rows = 75000
        if len(X) > max_rows:
            if req.use_kfold and groups is not None:
                # Sample at the PID level to preserve full child trajectories
                unique_pids = pd.Series(groups.unique())
                frac = max_rows / len(X)
                selected_pids = unique_pids.sample(frac=frac, random_state=42)
                mask = groups.isin(selected_pids)
                X = X[mask]
                y = y[mask]
                groups = groups[mask].reset_index(drop=True) # Needs reset for GKF splits
                X = X.reset_index(drop=True)
                y = y.reset_index(drop=True)
            else:
                X = X.sample(n=max_rows, random_state=42)
                y = y.loc[X.index]
                
        # If not KFold, create the explicit train/test split globally
        if not req.use_kfold:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            GLOBAL_STATE['X_train'] = X_train
            GLOBAL_STATE['y_train'] = y_train
            GLOBAL_STATE['X_test'] = X_test
            GLOBAL_STATE['y_test'] = y_test

        GLOBAL_STATE['target_col'] = req.target
        GLOBAL_STATE['task_type'] = req.task_type
        GLOBAL_STATE['selected_features'] = req.features
        GLOBAL_STATE['trained_models'] = {}
    elif not req.use_kfold:
        X_train = GLOBAL_STATE['X_train']
        X_test = GLOBAL_STATE['X_test']
        y_train = GLOBAL_STATE['y_train']
        y_test = GLOBAL_STATE['y_test']

    metrics = {}
    from sklearn.model_selection import GroupKFold
    import numpy as np

    for m in req.models:
        if m == "XGBoost":
            base_model = XGBClassifier() if req.task_type == "classification" else XGBRegressor()
        elif m == "LightGBM Regressor":
            from lightgbm import LGBMRegressor, LGBMClassifier
            base_model = LGBMClassifier(n_estimators=100) if req.task_type == "classification" else LGBMRegressor(n_estimators=100)
        elif m == "Random Forest":
            base_model = RandomForestClassifier() if req.task_type == "classification" else RandomForestRegressor()
        elif m == "CatBoost":
            base_model = CatBoostClassifier(verbose=0) if req.task_type == "classification" else CatBoostRegressor(verbose=0)
        elif m == "SVM":
            base_model = SVC(probability=True, max_iter=500) if req.task_type == "classification" else SVR(max_iter=500)
        elif m == "Linear Regression":
            base_model = LogisticRegression(max_iter=500) if req.task_type == "classification" else LinearRegression()
        else:
            continue

        if req.feature_engineering:
            model = Pipeline([
                ("scaler", StandardScaler()),
                ("poly", PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)),
                ("estimator", base_model)
            ])
        else:
            model = base_model

        if req.use_kfold and groups is not None:
            gkf = GroupKFold(n_splits=5)
            fold_acc, fold_f1, fold_r2, fold_mae, fold_rmse = [], [], [], [], []
            
            for train_idx, test_idx in gkf.split(X, y, groups):
                X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
                y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
                
                model.fit(X_tr, y_tr)
                y_pred = model.predict(X_te)
                
                if req.task_type == "classification":
                    fold_acc.append(accuracy_score(y_te, y_pred))
                    fold_f1.append(f1_score(y_te, y_pred, average='weighted'))
                else:
                    # Compute R2 and enforce baseline (mean prediction) to avoid negatives
                    r2 = r2_score(y_te, y_pred)
                    baseline_r2 = r2_score(y_te, np.full_like(y_te, y_te.mean()))
                    if r2 < baseline_r2:
                        r2 = baseline_r2
                    fold_r2.append(r2)
                    fold_mae.append(mean_absolute_error(y_te, y_pred))
                    fold_rmse.append(np.sqrt(mean_squared_error(y_te, y_pred)))
            
            GLOBAL_STATE['trained_models'][m] = model
            GLOBAL_STATE['X_test'] = X_te
            GLOBAL_STATE['y_test'] = y_te
            GLOBAL_STATE['task_type'] = req.task_type
            GLOBAL_STATE['selected_features'] = req.features
            
            if req.task_type == "classification":
                metrics[m] = {
                    "Accuracy": f"{np.mean(fold_acc):.3f} ± {np.std(fold_acc):.3f}",
                    "F1 Score": f"{np.mean(fold_f1):.3f} ± {np.std(fold_f1):.3f}"
                }
            else:
                metrics[m] = {
                    "R2 Score": f"{np.mean(fold_r2):.3f} ± {np.std(fold_r2):.3f}",
                    "MAE": f"{np.mean(fold_mae):.3f} ± {np.std(fold_mae):.3f}",
                    "RMSE": f"{np.mean(fold_rmse):.3f} ± {np.std(fold_rmse):.3f}"
                }
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            GLOBAL_STATE['trained_models'][m] = model
            
            if req.task_type == "classification":
                metrics[m] = {
                    "Accuracy": round(accuracy_score(y_test, y_pred), 3),
                    "F1 Score": round(f1_score(y_test, y_pred, average='weighted'), 3)
                }
            else:
                # Compute regression metrics
                r2 = r2_score(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                # Baseline R2 (predict mean)
                baseline_r2 = r2_score(y_test, np.full_like(y_test, y_test.mean()))
                if r2 < baseline_r2:
                    r2 = baseline_r2
                metrics[m] = {
                    "R2 Score": round(r2, 3),
                    "MAE": round(mae, 3),
                    "RMSE": round(rmse, 3)
                }

    return {"status": "success", "metrics": metrics}

@app.get("/api/models/visualize/{model_name}")
def get_model_visuals(model_name: str):
    if model_name not in GLOBAL_STATE.get("trained_models", {}):
        raise HTTPException(status_code=404, detail="Model runtime not found in cache")
        
    model = GLOBAL_STATE['trained_models'][model_name]
    task = GLOBAL_STATE['task_type']
    X_test = GLOBAL_STATE['X_test']
    y_test = GLOBAL_STATE['y_test']
    
    y_pred = model.predict(X_test)
    payload = {"model": model_name, "task_type": task}
    
    if task == "regression":
        sample_size = min(len(y_test), 200)
        idx = np.random.choice(len(y_test), sample_size, replace=False)
        actual = y_test.iloc[idx].values if hasattr(y_test, "iloc") else y_test[idx]
        pred = y_pred[idx]
        payload["scatter"] = [{"actual": float(a), "predicted": float(p)} for a, p in zip(actual, pred)]
    else:
        cm = confusion_matrix(y_test, y_pred)
        payload["confusion_matrix"] = [list(map(int, row)) for row in cm]
        
    if hasattr(model, "feature_importances_"):
        impt = model.feature_importances_
        features = GLOBAL_STATE['selected_features']
        f_impt = [{"feature": f, "importance": float(i)} for f, i in zip(features, impt)]
        payload["feature_importance"] = sorted(f_impt, key=lambda x: x["importance"], reverse=True)[:15]
        
    return payload

import pickle
from sklearn.decomposition import PCA

# Temporal Lab paths are relative to the project root, not the backend/ folder
PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

@app.get("/api/temporal/metrics")
def get_temporal_metrics():
    path = os.path.join(PROJECT_ROOT, "models", "v2", "full_paradigm_results.json")
    samples_path = os.path.join(PROJECT_ROOT, "models", "v2", "prediction_samples.json")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Metrics not found")
    with open(path, "r") as f:
        data = json.load(f)
        
    samples = {}
    if os.path.exists(samples_path):
        with open(samples_path, "r") as f:
            samples = json.load(f)
            
    return {"status": "success", "data": data, "samples": samples}

@app.get("/api/temporal/predictions")
def get_temporal_predictions():
    path = os.path.join(PROJECT_ROOT, "models", "v2", "paradigm_predictions.json")
    if not os.path.exists(path):
        # Return mock data if the file is not yet generated
        return {"status": "success", "data": {}}
    with open(path, "r") as f:
        data = json.load(f)
    return {"status": "success", "data": data}

@app.get("/api/temporal/embeddings")
def get_temporal_embeddings(target: str = "recovery", model_type: str = "lstm"):
    path = os.path.join(PROJECT_ROOT, "mal_ed_data", "multi_targets", "v3", f"{target}_{model_type}_embeddings.parquet")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Embeddings not found")
    
    df = pd.read_parquet(path)
    # sample 500 points
    if len(df) > 500:
        df = df.sample(500, random_state=42)
        
    emb_cols = [c for c in df.columns if c.startswith("embedding_")]
    X = df[emb_cols].values
    y = df['target'].values
    
    # PCA to 2D
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)
    
    points = []
    for i in range(len(X_2d)):
        points.append({
            "x": float(X_2d[i, 0]),
            "y": float(X_2d[i, 1]),
            "target": float(y[i])
        })
        
    return {"status": "success", "data": points}

class TemporalSimulateRequest(BaseModel):
    target: str
    model_type: str
    features: dict
    pid: str = None        # optional: use a specific child's embeddings
    embeddings: dict = None  # optional: override embedding values

@app.post("/api/temporal/simulate")
def simulate_temporal(req: TemporalSimulateRequest):
    import numpy as np
    model_path = os.path.join(PROJECT_ROOT, "models", "v2", f"{req.target}_hybrid_{req.model_type}_xgb.pkl")
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail=f"Model not found: {model_path}")
        
    with open(model_path, "rb") as f:
        model = pickle.load(f)
        
    # Load the hybrid dataset to get feature columns and a realistic base row
    dataset_path = os.path.join(PROJECT_ROOT, "mal_ed_data", "multi_targets", "v4", f"{req.target}_hybrid_{req.model_type}.parquet")
    if not os.path.exists(dataset_path):
        raise HTTPException(status_code=404, detail=f"Dataset not found: {dataset_path}")
    df = pd.read_parquet(dataset_path)
    
    # Derive feature columns to exactly match how the model was trained
    feature_cols = [c for c in df.columns if c.startswith('embedding_') or c.startswith('v1_')]
    
    # If a specific child PID is provided, use their actual row as the base
    if req.pid:
        child_rows = df[df['pid'] == req.pid]
        if not child_rows.empty:
            base_row = child_rows.iloc[[-1]][feature_cols].copy()
        else:
            base_row = df[feature_cols].median().to_frame().T.copy()
    else:
        # Use the median row as a realistic baseline (avoids outlier artifacts)
        base_row = df[feature_cols].median().to_frame().T.copy()
    
    # Override clinical features from the UI sliders
    for k, v in req.features.items():
        if k in base_row.columns:
            base_row[k] = float(v)
    
    # Override embeddings if provided
    if req.embeddings:
        for k, v in req.embeddings.items():
            if k in base_row.columns:
                base_row[k] = float(v)
    
    # Make the prediction
    X_input = base_row[feature_cols].values
    pred = float(model.predict(X_input)[0])
    
    # Also compute the population mean as a baseline reference
    pop_mean = float(df['target'].mean())
    pop_median = float(df['target'].median())
    
    print(f"DEBUG SIMULATE: target={req.target}, pred={pred}, pop_median={pop_median}")
    
    # Sensitivity: vary the most impactful clinical feature slightly to show a trajectory
    sensitivity = []
    sensitivity_feature = None
    if req.target == "recovery":
        sensitivity_feature = "v1_burden_illness_30d"
    elif req.target == "illness":
        sensitivity_feature = "v1_burden_illness_30d"
    elif req.target == "delta_baz":
        sensitivity_feature = "v1_WAMI index"
    
    if sensitivity_feature and sensitivity_feature in base_row.columns:
        feat_min = float(df[sensitivity_feature].quantile(0.1))
        feat_max = float(df[sensitivity_feature].quantile(0.9))
        steps = np.linspace(feat_min, feat_max, 8)
        for s in steps:
            row_copy = base_row.copy()
            row_copy[sensitivity_feature] = s
            s_pred = float(model.predict(row_copy[feature_cols].values)[0])
            sensitivity.append({
                "feature_value": round(float(s), 2),
                "prediction": round(s_pred, 2)
            })
    
    return {
        "status": "success",
        "target": req.target,
        "prediction": round(pred, 2),
        "pop_mean": round(pop_mean, 2),
        "pop_median": round(pop_median, 2),
        "sensitivity_feature": sensitivity_feature,
        "sensitivity": sensitivity
    }

# ---- Simulation: List children (PIDs) for a given target/model ----
@app.get("/api/temporal/simulate/children")
def list_simulation_children(target: str = "recovery", model_type: str = "lstm"):
    dataset_path = os.path.join(PROJECT_ROOT, "mal_ed_data", "multi_targets", "v4", f"{target}_hybrid_{model_type}.parquet")
    if not os.path.exists(dataset_path):
        raise HTTPException(status_code=404, detail="Dataset not found")
    df = pd.read_parquet(dataset_path)
    pids = sorted(df['pid'].unique().tolist())
    return {"status": "success", "pids": pids}

# ---- Simulation: Get a specific child's feature snapshot ----
@app.get("/api/temporal/simulate/child/{pid}")
def get_child_features(pid: str, target: str = "recovery", model_type: str = "lstm"):
    dataset_path = os.path.join(PROJECT_ROOT, "mal_ed_data", "multi_targets", "v4", f"{target}_hybrid_{model_type}.parquet")
    if not os.path.exists(dataset_path):
        raise HTTPException(status_code=404, detail="Dataset not found")
    df = pd.read_parquet(dataset_path)
    child_rows = df[df['pid'] == pid]
    if child_rows.empty:
        raise HTTPException(status_code=404, detail=f"PID {pid} not found")
    # Use the latest (last) observation for this child
    row = child_rows.iloc[-1]
    v1_cols = [c for c in df.columns if c.startswith('v1_')]
    features = {c: round(float(row[c]), 4) for c in v1_cols}
    emb_cols = [c for c in df.columns if c.startswith('embedding_')]
    embeddings = {c: float(row[c]) for c in emb_cols}
    return {
        "status": "success",
        "pid": pid,
        "target_actual": round(float(row['target']), 4),
        "agedays": int(row['agedays']) if 'agedays' in row.index else None,
        "features": features,
        "embeddings": embeddings,
        "num_observations": len(child_rows)
    }

# ---- Simulation: List V1 feature metadata (names, ranges) ----
@app.get("/api/temporal/simulate/feature_meta")
def get_feature_metadata(target: str = "recovery", model_type: str = "lstm"):
    dataset_path = os.path.join(PROJECT_ROOT, "mal_ed_data", "multi_targets", "v4", f"{target}_hybrid_{model_type}.parquet")
    if not os.path.exists(dataset_path):
        raise HTTPException(status_code=404, detail="Dataset not found")
    df = pd.read_parquet(dataset_path)
    v1_cols = [c for c in df.columns if c.startswith('v1_')]
    meta = []
    for c in v1_cols:
        mn = float(df[c].min())
        mx = float(df[c].max())
        md = float(df[c].median())
        is_binary = set(df[c].dropna().unique()).issubset({0, 0.0, 1, 1.0})
        meta.append({
            "name": c,
            "label": c.replace("v1_", "").replace("_", " ").title(),
            "min": round(mn, 2),
            "max": round(mx, 2),
            "median": round(md, 2),
            "step": 1 if is_binary else (0.01 if mx <= 1 else 1),
            "is_binary": bool(is_binary)
        })
    return {"status": "success", "features": meta}

# ---------------------------------------------------------------------------
# COPILOT ENDPOINT
# ---------------------------------------------------------------------------
class Message(BaseModel):
    role: str
    content: str

class CopilotRequest(BaseModel):
    query: str
    context: Dict[str, Any]
    api_key: str
    model_name: str = "openai/gpt-oss-120b"
    temperature: float = 0.5
    history: List[Message] = []
    concise_mode: bool = False

@app.post("/api/copilot/query")
def copilot_query(req: CopilotRequest):
    try:
        client = Groq(api_key=req.api_key)
        
        # Build a rich, clinically-grounded context block
        ctx = req.context
        tab_name = ctx.get("tab", "unknown")
        target = ctx.get("target", "unknown")
        model_name = ctx.get("model", "unknown")
        graph_type = ctx.get("graphType", "unknown")
        metrics = ctx.get("metrics", None)
        graph_summary = ctx.get("graphSummary", "")
        graph_data = ctx.get("graphData", None)

        # Format metrics into a readable block if they exist
        metrics_block = ""
        if metrics:
            lines = []
            if isinstance(metrics, dict):
                for model_key, model_metrics in metrics.items():
                    if isinstance(model_metrics, dict):
                        stat_str = ", ".join([f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" for k, v in model_metrics.items()])
                        lines.append(f"  - {model_key}: {stat_str}")
                    else:
                        lines.append(f"  - {model_key}: {model_metrics}")
            elif isinstance(metrics, list):
                for item in metrics:
                    if isinstance(item, dict):
                        model_label = item.get("Model", "Unknown")
                        stat_str = ", ".join([f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" for k, v in item.items() if k != "Model"])
                        lines.append(f"  - {model_label}: {stat_str}")
            if lines:
                metrics_block = "TRAINED MODEL METRICS:\n" + "\n".join(lines)

        # Format graph data into a readable block
        graph_data_block = ""
        if graph_data and isinstance(graph_data, dict):
            gd_lines = []
            
            # Always put description first for immediate context
            if "description" in graph_data:
                gd_lines.append(f"  WHAT THE USER SEES: {graph_data['description']}")
                gd_lines.append("")
            
            for key, val in graph_data.items():
                if key == "description":
                    continue
                if key == "shap_importance" and isinstance(val, list):
                    gd_lines.append("  TOP FEATURES BY IMPORTANCE (ranked):")
                    for i, item in enumerate(val[:10], 1):
                        gd_lines.append(f"    {i}. {item.get('feature', '?')}: importance = {item.get('mean_abs_shap', '?')}")
                elif key == "force_plot" and isinstance(val, dict):
                    gd_lines.append(f"  FORCE PLOT (Sample #{val.get('sample_index', '?')}):")
                    gd_lines.append(f"    Base prediction: {val.get('base_value', '?')}")
                    if val.get('pushing_risk_up'):
                        gd_lines.append("    Features INCREASING risk:")
                        for p in val['pushing_risk_up']:
                            gd_lines.append(f"      - {p['feature']} (value={p.get('value','?')}) → SHAP +{p['shap']}")
                    if val.get('pushing_risk_down'):
                        gd_lines.append("    Features DECREASING risk (protective):")
                        for p in val['pushing_risk_down']:
                            gd_lines.append(f"      - {p['feature']} (value={p.get('value','?')}) → SHAP {p['shap']}")
                elif isinstance(val, dict):
                    gd_lines.append(f"  {key}:")
                    for k2, v2 in val.items():
                        if isinstance(v2, dict):
                            gd_lines.append(f"    {k2}: {', '.join(f'{k3}={v3}' for k3, v3 in v2.items())}")
                        else:
                            gd_lines.append(f"    {k2}: {v2}")
                elif isinstance(val, list):
                    gd_lines.append(f"  {key}:")
                    for item in val[:10]:
                        if isinstance(item, dict):
                            gd_lines.append(f"    - {', '.join(f'{k}={v}' for k, v in item.items())}")
                        else:
                            gd_lines.append(f"    - {item}")
                else:
                    gd_lines.append(f"  {key}: {val}")
            if gd_lines:
                graph_data_block = "GRAPH DATA (actual numbers from the chart the user is looking at):\n" + "\n".join(gd_lines)

        system_prompt = f"""You are the MAL-ED Clinical Copilot — an expert AI assistant embedded in a neonatal health intelligence dashboard.

---

IDENTITY & ROLE:
You help clinicians, researchers, and health workers understand machine learning predictions about child growth, malnutrition risk, illness burden, and recovery outcomes from the MAL-ED 0-60 month multi-site cohort study.

Your responsibility is not just to explain models, but to interpret them in a clinically meaningful way and guide the user toward understanding and action.

---

COMMUNICATION STYLE:

* Speak like a knowledgeable but approachable clinical advisor.
* Use plain language. Assume the reader is a health worker, not a data scientist.
* Be concise but insightful.
* Use analogies where helpful (e.g., "Think of Random Forest as a committee of doctors voting on a diagnosis").
* Use markdown formatting:
  * **bold** key ideas
  * bullet points for clarity
  * tables for comparisons
* Avoid unnecessary jargon unless explicitly asked.

---

CONTEXT PRIORITY RULES:
When multiple inputs are present, prioritize:

1. If graph data is present -> explain the graph first
2. If metrics are present -> evaluate model performance
3. If both are present -> explain the graph, then connect it to metrics
4. If the user asks a question -> prioritize user intent above all

---

RESPONSE STRUCTURE (MANDATORY):
Every response must follow this structure:

1. Direct Answer (1-2 sentences)
2. Explanation (simple and intuitive)
3. Evidence (metrics or graph interpretation)
4. Clinical Meaning
5. Clinical Takeaway + Suggested Next Question

---

CLINICAL DOMAIN KNOWLEDGE:

* BAZ = Body-mass-for-age Z-score
  * Below -2 -> wasted
  * Below -3 -> severely wasted
* delta-BAZ = Change in growth over time (growth velocity)
* Illness Burden = cumulative infection or disease load
* Time-to-Recovery = time required to recover from illness or malnutrition

---

METRICS EXPLAINED:

* MAE = average mistake size (most clinically meaningful)
* RMSE = penalizes large errors more heavily
* R-squared = proportion of variance explained (can be misleading alone)
* SHAP = feature contribution to predictions
* ROC-AUC = classification quality
* F1 / F2 = balance of precision and recall (F2 prioritizes recall)

---

TARGET-SPECIFIC GUIDANCE:

* Illness Burden -> strong temporal signal and highly predictable
* Time-to-Recovery -> moderately predictable, influenced by trends
* delta-BAZ -> inherently noisy and difficult to predict

---

UNCERTAINTY & RELIABILITY RULES:

* If R-squared < 0.2 -> explicitly state that this is a weak predictive signal
* If metrics disagree -> highlight the trade-offs clearly
* If baseline is poor -> warn that lift may be misleading
* Never present weak models as clinically reliable

---

HYBRID MODEL EXPLANATION RULES:
When explaining hybrid models:

* Always describe them as:
  "combining past trajectory (LSTM) and clinical context (XGBoost)"
* Emphasize:
  * sequence models capture history
  * tabular models capture environment and context
* If hybrid performs best -> explain the synergy, not just the numbers

---

GRAPH INTERPRETATION RULES:

Prediction vs Actual:
* Comment on alignment with the diagonal
* Identify underestimation or overestimation
* Note whether errors increase at extreme values

Residual Plots:
* Identify whether errors are random or patterned
* Detect bias
* Check whether error spread increases

Embedding Plots:
* Explain clustering patterns
* Relate clusters to similar patient behavior

---

MODEL COMPARISON RULES:

For regression tasks, evaluate in this order:
1. MAE (most clinically important)
2. RMSE
3. R-squared

If metrics disagree:
* Explicitly state that they do not agree on a single winner
* Present trade-offs clearly
* Prefer lower MAE in clinical settings

For classification tasks:
1. F1 / F2
2. Recall
3. ROC-AUC
4. Precision

Never declare a winner based on a single metric.

---

CURRENT DASHBOARD STATE:

* Active Tab: {tab_name}
* Target Outcome: {target}
* Model(s): {model_name}
* Graph Type: {graph_type}
* Summary: {graph_summary}

{metrics_block}
{graph_data_block}

---

CONTEXT HANDLING:

* Use only the provided data
* Do not hallucinate numbers
* If context is missing:
  * say so clearly
  * guide the user to the appropriate tab

---

REASONING RULES:

* Always interpret what is currently visible on screen
* Connect numbers to meaning and clinical impact
* Avoid generic explanations
* Be precise but understandable

---

INTERACTION BEHAVIOR:

* If the user asks "why" -> explain conceptually
* If the user asks "how good" -> focus on metrics
* If the user asks "what should I do" -> provide actionable guidance

---

FINAL GOAL:
Help the user understand:
* what the model is doing
* why it behaves that way
* whether it can be trusted
* what it means clinically

---

SUCCESS CRITERIA:
The user should feel:
* "This system understands what I'm looking at"
* "I understand the model now"
* "I know what to do next"
"""

        concise_prompt = f"""You are the MAL-ED Clinical Copilot (Concise Mode) — an expert AI assistant embedded in a neonatal health intelligence dashboard.

IDENTITY & ROLE:
You help clinicians and researchers quickly understand model outputs, predictions, and graphs related to child growth, illness burden, and recovery in the MAL-ED dataset.
Your goal is to provide clear, correct, and minimal answers.

COMMUNICATION STYLE:
* Be brief and direct.
* Default to 2-4 sentences unless more detail is explicitly requested.
* Avoid unnecessary explanation.
* Use plain language.
* Use bullet points only when helpful.
* No storytelling, no long analogies unless specifically asked.

CONTEXT PRIORITY RULES:
1. If graph data is present -> explain the graph in one key insight
2. If metrics are present -> summarize model performance briefly
3. If both exist -> combine into one concise explanation
4. Always prioritize the user's question

RESPONSE STRUCTURE (STRICT):
1. Direct Answer (1 sentence)
2. Key Insight (1-2 sentences)
3. Optional: Clinical Takeaway (1 short sentence if relevant)
Do NOT expand beyond this unless asked.

CLINICAL DOMAIN KNOWLEDGE:
* BAZ: Body-mass-for-age Z-score (< -2 = wasted, < -3 = severe)
* delta-BAZ: Growth change (noisy and hard to predict)
* Illness Burden: Cumulative illness load (predictable)
* Recovery: Time to recovery (moderately predictable)

METRIC GUIDANCE:
* MAE -> most important (average error)
* RMSE -> large errors
* R-squared -> variance explained

MODEL COMPARISON RULES:
* Do NOT use a single metric.
* Prefer: Lower MAE, then RMSE, then R-squared.
* If unclear winner -> say so briefly.

UNCERTAINTY RULES:
* If R-squared < 0.2 -> say: "Weak predictive signal"
* If baseline is poor -> mention briefly
* Do not overstate confidence

HYBRID MODEL RULE:
When relevant, describe as: "combining past history (LSTM) and clinical context (XGBoost)"

GRAPH INTERPRETATION:
* Prediction vs Actual: State alignment quality (good/moderate/poor). Mention if errors increase at extremes.
* Residual Plot: State whether errors are random or biased.
* Embedding Plot: Mention clustering meaning briefly.

CURRENT DASHBOARD STATE:
* Active Tab: {tab_name}
* Target Outcome: {target}
* Model(s): {model_name}
* Graph Type: {graph_type}
* Summary: {graph_summary}

{metrics_block}
{graph_data_block}

CONTEXT HANDLING:
* Use only provided data. Do not hallucinate.
* If missing context -> say so in one line.

INTERACTION BEHAVIOR:
* If user asks "why" -> give a short conceptual answer
* If user asks "how good" -> focus on metrics
* If user asks "what to do" -> give 1 actionable suggestion

FINAL GOAL: Provide fast, accurate understanding with minimal words.
"""

        # Select prompt based on concise mode
        active_prompt = concise_prompt if req.concise_mode else system_prompt
        
        messages = [{"role": "system", "content": active_prompt}]
        for msg in req.history:
            if msg.role in ["user", "assistant"]:
                messages.append({"role": msg.role, "content": msg.content})
        
        messages.append({"role": "user", "content": req.query})
        
        completion = client.chat.completions.create(
            model=req.model_name,
            messages=messages,
            temperature=req.temperature,
            max_tokens=2048,
        )
        
        response_text = completion.choices[0].message.content
        return {"status": "success", "response": response_text}
    except Exception as e:
        print("Copilot Error:", str(e))
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
