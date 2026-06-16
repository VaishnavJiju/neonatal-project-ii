import os
import json
import joblib
import pandas as pd
import numpy as np
import shap
from pathlib import Path
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.dummy import DummyRegressor, DummyClassifier
from xgboost import XGBRegressor, XGBClassifier
from catboost import CatBoostRegressor, CatBoostClassifier
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, roc_auc_score, f1_score, accuracy_score, precision_score, recall_score, fbeta_score

TARGETS = {
    "baz_ar": {"file": "dataset_baz_ar.parquet", "col": "target", "task": "regression"},
    "delta_baz": {"file": "dataset_baz_delta.parquet", "col": "target_delta", "task": "regression"},
    "diarrhea": {"file": "dataset_diarrhea_v1_full.parquet", "col": "classification_target", "task": "classification"},
    "illness_burden": {"file": "dataset_illness_burden.parquet", "col": "burden_target", "task": "regression"}
}

MODELS_REGRESSION = {
    "Random Forest": RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1),
    "XGBoost": XGBRegressor(n_estimators=50, max_depth=6, random_state=42, n_jobs=-1),
    "CatBoost": CatBoostRegressor(iterations=50, depth=6, random_state=42, verbose=0)
}

MODELS_CLASSIFICATION = {
    "Random Forest": RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1, class_weight='balanced'),
    "XGBoost": XGBClassifier(n_estimators=50, max_depth=6, random_state=42, n_jobs=-1),
    "CatBoost": CatBoostClassifier(iterations=50, depth=6, random_state=42, verbose=0, auto_class_weights='Balanced')
}

def create_directory(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def precompute_v1_registry():
    print("Initializing V1 Model Registry Precomputation...")
    base_dir = "mal_ed_data/multi_targets/v1"
    registry_dir = "models/v1"
    v0_registry_dir = "models/v0"
    
    for target_key, config in TARGETS.items():
        print(f"\n--- Processing V1 Target: {target_key} ---")
        filepath = os.path.join(base_dir, config["file"])
        if not os.path.exists(filepath):
            print(f"Skipping {target_key}. File not found: {filepath}")
            continue
            
        df = pd.read_parquet(filepath)
        target_col = config["col"]
        df = df.dropna(subset=[target_col])
        
        groups = df['pid']
        X = df.drop(columns=["pid", target_col])
        
        # Numeric coercion & Column Sanitation
        for c in X.columns:
            if X[c].dtype.name == 'category' or X[c].dtype == 'object':
                X[c] = X[c].astype(str).str.lower().str.strip()
                X[c] = X[c].map({"yes": 1, "no": 0, "1": 1, "0": 0, "missing": -1, "true": 1, "false": 0}).fillna(-1)
            else:
                X[c] = pd.to_numeric(X[c], errors='coerce')
                X[c] = X[c].fillna(X[c].median())
        
        X.columns = [c.replace('[', '_').replace(']', '_').replace('<', '_') for c in X.columns]
        
            
        if config["task"] == "classification":
            y = df[target_col].astype(int)
            models = MODELS_CLASSIFICATION
            dummy = DummyClassifier(strategy="prior")
        else:
            y = pd.to_numeric(df[target_col], errors='coerce')
            models = MODELS_REGRESSION
            dummy = DummyRegressor(strategy="mean")
            
        target_dir = os.path.join(registry_dir, target_key)
        create_directory(target_dir)
        
        gkf = GroupKFold(n_splits=5)
        splits = list(gkf.split(X, y, groups=groups))
        
        print("Calculating baseline metrics...")
        baseline_preds = cross_val_predict(dummy, X, y, cv=splits, groups=groups)
        
        if config["task"] == "regression":
            baseline_metrics = {
                "R2": r2_score(y, baseline_preds),
                "MAE": mean_absolute_error(y, baseline_preds),
                "RMSE": np.sqrt(mean_squared_error(y, baseline_preds))
            }
            primary_metric = "R2"
        else:
            baseline_metrics = {
                "ROC-AUC": 0.5,
                "F1": f1_score(y, baseline_preds, average='weighted', zero_division=0),
                "F2": fbeta_score(y, baseline_preds, beta=2, average='weighted', zero_division=0)
            }
            primary_metric = "ROC-AUC"
            
        for model_name, model in models.items():
            model_slug = model_name.lower().replace(" ", "_")
            model_dir = os.path.join(target_dir, model_slug)
            
            if os.path.exists(os.path.join(model_dir, "metadata.json")):
                print(f"Skipping {model_name} (already computed).")
                continue
                
            print(f" -> Training {model_name}...")
            cv_preds = cross_val_predict(model, X, y, cv=splits, groups=groups)
            cv_probs = None
            if config["task"] == "classification" and hasattr(model, 'predict_proba'):
                cv_probs = cross_val_predict(model, X, y, cv=splits, groups=groups, method='predict_proba')[:, 1]
            
            metrics = {}
            if config["task"] == "regression":
                metrics = {
                    "R2": r2_score(y, cv_preds),
                    "MAE": mean_absolute_error(y, cv_preds),
                    "RMSE": np.sqrt(mean_squared_error(y, cv_preds)),
                    "Lift (R2)": r2_score(y, cv_preds) - baseline_metrics["R2"]
                }
            else:
                auc = roc_auc_score(y, cv_probs) if cv_probs is not None else 0.5
                metrics = {
                    "ROC-AUC": auc,
                    "F1": f1_score(y, cv_preds, average='weighted', zero_division=0),
                    "F2": fbeta_score(y, cv_preds, beta=2, average='weighted', zero_division=0),
                    "Lift (AUC)": auc - baseline_metrics["ROC-AUC"]
                }
            
            model.fit(X, y)
            
            feature_importance = {}
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                feature_importance = {f: float(i) for f, i in zip(X.columns, importances)}
                
            create_directory(model_dir)
            joblib.dump(model, os.path.join(model_dir, "model.pkl"))
            
            meta = {
                "version": "v1",
                "target": target_key,
                "model_name": model_name,
                "task": config["task"],
                "features": list(X.columns),
                "metrics": metrics,
                "baseline": baseline_metrics
            }
            with open(os.path.join(model_dir, "metadata.json"), "w") as f:
                json.dump(meta, f, indent=2)
                
            with open(os.path.join(model_dir, "feature_importance.json"), "w") as f:
                json.dump(feature_importance, f, indent=2)
                
            pred_df = pd.DataFrame({"y_true": y, "y_pred": cv_preds})
            if cv_probs is not None: pred_df["y_prob"] = cv_probs
            pred_df.to_parquet(os.path.join(model_dir, "predictions.parquet"))
            
            # --- SHAP CORE ---
            sample_size = min(300, len(X))
            idx = np.random.RandomState(42).choice(len(X), sample_size, replace=False)
            X_sample = X.iloc[idx].copy()
            
            print(f"   => Computing SHAP values for {model_name}...")
            if "XGB" in str(type(model)):
                try: model.get_booster().set_param({'base_score': 0.5})
                except: pass
                
            try:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_sample)
                expected_val = explainer.expected_value
            except Exception as e:
                print(f"TreeExplainer failed: {e}. Falling back to Linear/Permutation...")
                explainer = shap.Explainer(model.predict, X_sample)
                shap_values = explainer(X_sample).values
                expected_val = explainer.expected_value if hasattr(explainer, 'expected_value') else y.mean()
            
            if isinstance(shap_values, list):
                shap_values_to_save = shap_values[1]
                ev_to_save = expected_val[1] if isinstance(expected_val, (list, np.ndarray)) else expected_val
            elif len(shap_values.shape) == 3:
                shap_values_to_save = shap_values[:, :, 1]
                ev_to_save = expected_val[1] if isinstance(expected_val, (list, np.ndarray)) and len(expected_val) > 1 else expected_val
            else:
                shap_values_to_save = shap_values
                ev_to_save = expected_val[0] if isinstance(expected_val, (list, np.ndarray)) else expected_val
                
            mean_abs_shap = np.abs(shap_values_to_save).mean(axis=0)
            global_shap = {f: float(val) for f, val in zip(X.columns, mean_abs_shap)}
            
            # Create a combined dataframe for Beeswarm: SHAP values + Feature Values
            shap_df = pd.DataFrame(shap_values_to_save, columns=[f"shap_{c}" for c in X.columns])
            val_df = X_sample.reset_index(drop=True)
            val_df.columns = [f"val_{c}" for c in val_df.columns]
            combined_df = pd.concat([shap_df, val_df], axis=1)
            combined_df.to_parquet(os.path.join(model_dir, "shap_sample.parquet"))
            
            with open(os.path.join(model_dir, "shap_global.json"), "w") as f:
                json.dump(global_shap, f, indent=2)

            with open(os.path.join(model_dir, "expected_value.json"), "w") as f:
                json.dump({"expected_value": float(ev_to_save)}, f, indent=2)

            # --- POST-TRAINING VALIDATION COMPARISON (v0 vs v1) ---
            v0_meta_path = os.path.join(v0_registry_dir, target_key, model_slug, "metadata.json")
            if os.path.exists(v0_meta_path):
                with open(v0_meta_path, "r") as f:
                    v0_meta = json.load(f)
                diff = metrics[primary_metric] - v0_meta["metrics"].get(primary_metric, 0)
                comparison = {
                    "v0_model": v0_meta["metrics"],
                    "v1_model": metrics,
                    "accuracy_difference": diff,
                    "clinical_vs_raw_dominance": f"{primary_metric} shift of {diff:.4f}"
                }
                with open(os.path.join(model_dir, "comparison.json"), "w") as f:
                    json.dump(comparison, f, indent=2)

    print("\nV1 Registry Compiled and Certified!")

if __name__ == "__main__":
    precompute_v1_registry()
