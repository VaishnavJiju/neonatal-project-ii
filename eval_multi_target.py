import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, roc_auc_score, f1_score, accuracy_score, fbeta_score
import warnings
warnings.filterwarnings('ignore')

def clean_features(df, target_col):
    X = df.drop(columns=["pid", target_col]).copy()
    # Map unknown categorical variables and enforce numeric features
    for col in X.select_dtypes(include=[object]).columns:
        X[col] = X[col].replace({"Unknown": np.nan, "unknown": np.nan, "": np.nan})
        X[col] = pd.to_numeric(X[col], errors="coerce")
        median_val = X[col].median()
        if pd.isna(median_val): median_val = 0
        X[col] = X[col].fillna(median_val)

    for col in X.select_dtypes(include=['category']).columns:
        X[col] = X[col].cat.codes

    X = X.fillna(0)
    return X

def evaluate_regression(name, path, target_col, baseline_func):
    print(f"\n========================================")
    print(f"TASK: {name}")
    print(f"========================================")
    
    df = pd.read_parquet(path)
    # Drop rows where target is missing
    df = df.dropna(subset=[target_col])
    
    X = clean_features(df, target_col)
    y = df[target_col]
    groups = df["pid"]
    
    gkf = GroupKFold(n_splits=5)
    
    cv_r2, cv_mae, cv_rmse = [], [], []
    base_r2, base_mae, base_rmse = [], [], []
    
    for train_idx, val_idx in gkf.split(X, y, groups):
        rf = RandomForestRegressor(n_estimators=50, max_depth=10, max_samples=0.5, random_state=42, n_jobs=-1)
        rf.fit(X.iloc[train_idx], y.iloc[train_idx])
        
        preds = rf.predict(X.iloc[val_idx])
        y_val_actual = y.iloc[val_idx]
        
        # Calculate Base Prediction
        base_preds = baseline_func(df.iloc[val_idx])
        
        # Collect Metrics
        cv_r2.append(r2_score(y_val_actual, preds))
        cv_mae.append(mean_absolute_error(y_val_actual, preds))
        cv_rmse.append(np.sqrt(mean_squared_error(y_val_actual, preds)))
        
        base_r2.append(r2_score(y_val_actual, base_preds))
        base_mae.append(mean_absolute_error(y_val_actual, base_preds))
        base_rmse.append(np.sqrt(mean_squared_error(y_val_actual, base_preds)))

    r2_m, r2_s = np.mean(cv_r2), np.std(cv_r2)
    br2_m, br2_s = np.mean(base_r2), np.std(base_r2)

    mae_m, mae_s = np.mean(cv_mae), np.std(cv_mae)
    bmae_m, bmae_s = np.mean(base_mae), np.std(base_mae)
    
    rmse_m, rmse_s = np.mean(cv_rmse), np.std(cv_rmse)
    brmse_m, brmse_s = np.mean(base_rmse), np.std(base_rmse)

    print(f"R²   -> Baseline: {br2_m:.4f} | Model: {r2_m:.4f}   (Lift: {r2_m - br2_m:+.4f})")
    print(f"MAE  -> Baseline: {bmae_m:.4f} | Model: {mae_m:.4f}   (Lift: {bmae_m - mae_m:+.4f})")
    print(f"RMSE -> Baseline: {brmse_m:.4f} | Model: {rmse_m:.4f}   (Lift: {brmse_m - rmse_m:+.4f})")
    
def evaluate_classification(name, path, target_col, baseline_func):
    print(f"\n========================================")
    print(f"TASK: {name} (Classification)")
    print(f"========================================")
    
    df = pd.read_parquet(path)
    df = df.dropna(subset=[target_col])
    
    X = clean_features(df, target_col)
    y = df[target_col].astype(int)
    groups = df["pid"]
    
    gkf = GroupKFold(n_splits=5)
    
    cv_auc, cv_f1, cv_f2, cv_acc = [], [], [], []
    base_auc, base_f1, base_f2, base_acc = [], [], [], []
    
    for train_idx, val_idx in gkf.split(X, y, groups):
        rf = RandomForestClassifier(n_estimators=50, max_depth=10, max_samples=0.5, random_state=42, n_jobs=-1, class_weight='balanced')
        rf.fit(X.iloc[train_idx], y.iloc[train_idx])
        
        preds_proba = rf.predict_proba(X.iloc[val_idx])[:, 1]
        preds_class = rf.predict(X.iloc[val_idx])
        y_val_actual = y.iloc[val_idx]
        
        # Baseline (0 or 1 based on heuristics)
        base_preds_class = baseline_func(df.iloc[val_idx]).astype(int)
        
        cv_auc.append(roc_auc_score(y_val_actual, preds_proba))
        cv_f1.append(f1_score(y_val_actual, preds_class))
        cv_f2.append(fbeta_score(y_val_actual, preds_class, beta=2))
        cv_acc.append(accuracy_score(y_val_actual, preds_class))
        
        # Baseline AUC is purely 0.5 for constant/threshold baselines 
        # So we just evaluate accuracy and F1 for baselines
        base_f1.append(f1_score(y_val_actual, base_preds_class))
        base_f2.append(fbeta_score(y_val_actual, base_preds_class, beta=2))
        base_acc.append(accuracy_score(y_val_actual, base_preds_class))

    auc_m = np.mean(cv_auc)
    
    f1_m = np.mean(cv_f1)
    bf1_m = np.mean(base_f1)
    
    f2_m = np.mean(cv_f2)
    bf2_m = np.mean(base_f2)
    
    acc_m = np.mean(cv_acc)
    bacc_m = np.mean(base_acc)

    print(f"ROC-AUC -> Model: {auc_m:.4f}")
    print(f"F1      -> Baseline: {bf1_m:.4f} | Model: {f1_m:.4f}   (Lift: {f1_m - bf1_m:+.4f})")
    print(f"F2      -> Baseline: {bf2_m:.4f} | Model: {f2_m:.4f}   (Lift: {f2_m - bf2_m:+.4f})")
    print(f"Accuracy-> Baseline: {bacc_m:.4f} | Model: {acc_m:.4f}   (Lift: {acc_m - bacc_m:+.4f})")


if __name__ == "__main__":
    multi_targets_dir = "mal_ed_data/multi_targets"

    # 1. BAZ Autoregressive
    evaluate_regression(
        "BAZ Forecast (AR)", 
        f"{multi_targets_dir}/dataset_baz_ar.parquet", 
        "target",
        baseline_func=lambda d: d["target_prev"]
    )

    # 2. BAZ Delta
    evaluate_regression(
        "Delta BAZ (Change in Growth)", 
        f"{multi_targets_dir}/dataset_baz_delta.parquet", 
        "target_delta",
        baseline_func=lambda d: np.zeros(len(d))  # Baseline assumes no change (Delta = 0)
    )

    # 3. Near-Term Diarrhea classification
    # Baseline: If they had diarrhea today (ev_diarrhea), guess they have it soon. Else 0.
    evaluate_classification(
        "Near-Term Diarrhea Risk", 
        f"{multi_targets_dir}/dataset_diarrhea.parquet", 
        "classification_target",
        baseline_func=lambda d: d["ev_diarrhea"].fillna(0)
    )

    # 4. Illness Burden Forecast
    # Baseline: Assume the next window's burden == past window's burden
    evaluate_regression(
        "Illness Burden Forecast", 
        f"{multi_targets_dir}/dataset_illness_burden.parquet", 
        "burden_target",
        baseline_func=lambda d: d["burden_illness_30d"].fillna(0)
    )
