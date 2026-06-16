"""
Temporal Intelligence Lab — Phase 5: Full Paradigm Evaluation
=============================================================
Compares all model paradigms across all targets:
  Baseline, V1 (Feature), LSTM, TCN, Hybrid_LSTM, Hybrid_TCN

Metrics: R2, MAE, RMSE, Lift vs Baseline
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os, json, pickle
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor

# ============================================================
# SEQUENCE MODEL DEFINITIONS
# ============================================================
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc1 = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc2(self.relu(self.fc1(out[:, -1, :])))

class Chomp1d(nn.Module):
    def __init__(self, cs):
        super().__init__()
        self.cs = cs
    def forward(self, x):
        return x[:, :, :-self.cs].contiguous()

class TCNBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super().__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU(); self.dropout1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU(); self.dropout2 = nn.Dropout(dropout)
        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TCNModel(nn.Module):
    def __init__(self, input_size, nc=[64, 64], ks=3):
        super().__init__()
        layers = []
        for i in range(len(nc)):
            d = 2**i; ic = input_size if i == 0 else nc[i-1]; p = (ks-1)*d
            layers.append(TCNBlock(ic, nc[i], ks, 1, d, p))
        self.tcn = nn.Sequential(*layers)
        self.fc = nn.Linear(nc[-1], 1)
    def forward(self, x):
        x = x.transpose(1, 2)
        return self.fc(self.tcn(x)[:, :, -1])

# ============================================================
# CONFIG
# ============================================================
SEQ_DIR = 'mal_ed_data/multi_targets/v2'
MODEL_DIR = 'models/v2'
HYBRID_DIR = 'mal_ed_data/multi_targets/v4'
TARGETS = ['delta_baz', 'illness', 'recovery']

with open(os.path.join(SEQ_DIR, 'metadata.json'), 'r') as f:
    META = json.load(f)
SEQ_LEN = META['window_size']
N_FEATURES = META['features_per_timestep']

# ============================================================
# EVALUATE SEQUENCE MODELS (LSTM / TCN)
# ============================================================
def eval_sequence_model(target, model_type):
    df = pd.read_parquet(os.path.join(SEQ_DIR, f'{target}_sequences.parquet'))
    feat_cols = [f'f_{i}' for i in range(SEQ_LEN * N_FEATURES)]
    X, y, pids = df[feat_cols].values, df['y'].values, df['pid'].values
    
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    _, val_idx = next(gss.split(X, y, groups=pids))
    X_val, y_val = X[val_idx], y[val_idx]
    
    with open(os.path.join(MODEL_DIR, f'scaler_{target}.pkl'), 'rb') as f:
        scaler = pickle.load(f)
    X_scaled = scaler.transform(X_val.reshape(-1, N_FEATURES)).reshape(-1, SEQ_LEN, N_FEATURES)
    X_t = torch.tensor(X_scaled, dtype=torch.float32)
    
    model = LSTMModel(N_FEATURES) if model_type == 'lstm' else TCNModel(N_FEATURES)
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, f'{model_type}_{target}.pt'), map_location='cpu'))
    model.eval()
    
    with torch.no_grad():
        y_pred = model(X_t).numpy().flatten()
    
    return y_val, y_pred

# ============================================================
# EVALUATE HYBRID MODELS (XGBoost on embeddings + V1 features)
# ============================================================
def eval_hybrid_model(target, model_type):
    path = os.path.join(HYBRID_DIR, f'{target}_hybrid_{model_type}.parquet')
    if not os.path.exists(path):
        return None, None
    
    df = pd.read_parquet(path)
    feature_cols = [c for c in df.columns if c.startswith('embedding_') or c.startswith('v1_')]
    
    X = df[feature_cols].values
    y = df['target'].values
    pids = df['pid'].values
    
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(gss.split(X, y, groups=pids))
    
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    
    model = XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1, 
                         random_state=42, verbosity=0, n_jobs=-1)
    model.fit(X_train, y_train)
    
    # Save the model for API use
    with open(os.path.join(MODEL_DIR, f'{target}_hybrid_{model_type}_xgb.pkl'), 'wb') as f:
        pickle.dump(model, f)
        
    y_pred = model.predict(X_val)
    
    return y_val, y_pred

# ============================================================
# BASELINES
# ============================================================
def get_baseline(target, y_val, X_val_seq=None):
    if target == 'delta_baz':
        return np.zeros_like(y_val), "Zero Growth"
    elif target == 'illness':
        # illness_days is feature index 7 in last timestep
        idx = (SEQ_LEN - 1) * N_FEATURES + 7
        return X_val_seq[:, idx] if X_val_seq is not None else np.zeros_like(y_val), "Persistence"
    elif target == 'recovery':
        idx = (SEQ_LEN - 1) * N_FEATURES + 1
        return X_val_seq[:, idx] if X_val_seq is not None else np.zeros_like(y_val), "Last Interval"

def compute_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return {
        'R2': round(r2_score(y_true, y_pred), 4),
        'MAE': round(mean_absolute_error(y_true, y_pred), 4),
        'RMSE': round(np.sqrt(mse), 4),
    }

# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 100)
    print("TEMPORAL INTELLIGENCE LAB - Phase 5: Full Paradigm Evaluation")
    print("=" * 100)
    
    all_results = {}
    prediction_samples = {}
    
    for target in TARGETS:
        print(f"\n{'='*80}")
        print(f"TARGET: {target}")
        print(f"{'='*80}")
        
        results = []
        
        # Get sequence data for baseline computation
        df = pd.read_parquet(os.path.join(SEQ_DIR, f'{target}_sequences.parquet'))
        feat_cols = [f'f_{i}' for i in range(SEQ_LEN * N_FEATURES)]
        X_all, y_all, pids = df[feat_cols].values, df['y'].values, df['pid'].values
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        _, val_idx = next(gss.split(X_all, y_all, groups=pids))
        X_val_seq, y_val_seq = X_all[val_idx], y_all[val_idx]
        
        # 1. Baseline
        y_base, base_name = get_baseline(target, y_val_seq, X_val_seq)
        base_metrics = compute_metrics(y_val_seq, y_base)
        base_metrics['Model'] = f'Baseline ({base_name})'
        results.append(base_metrics)
        
        # 2. V1 Feature Model (XGBoost on raw V1 tabular)
        # Train a quick XGBoost on the sequence features as a "V1-like" baseline
        train_idx_seq = np.setdiff1d(np.arange(len(X_all)), val_idx)
        xgb_v1 = XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1,
                               random_state=42, verbosity=0, n_jobs=-1)
        xgb_v1.fit(X_all[train_idx_seq], y_all[train_idx_seq])
        y_pred_v1 = xgb_v1.predict(X_val_seq)
        v1_metrics = compute_metrics(y_val_seq, y_pred_v1)
        v1_metrics['Model'] = 'V1 (XGBoost Tabular)'
        results.append(v1_metrics)
        
        # 3. LSTM
        y_val_lstm, y_pred_lstm = eval_sequence_model(target, 'lstm')
        lstm_metrics = compute_metrics(y_val_lstm, y_pred_lstm)
        lstm_metrics['Model'] = 'LSTM'
        results.append(lstm_metrics)
        
        # 4. TCN
        y_val_tcn, y_pred_tcn = eval_sequence_model(target, 'tcn')
        tcn_metrics = compute_metrics(y_val_tcn, y_pred_tcn)
        tcn_metrics['Model'] = 'TCN'
        results.append(tcn_metrics)
        
        # 5. Hybrid LSTM
        y_val_hl, y_pred_hl = eval_hybrid_model(target, 'lstm')
        if y_val_hl is not None:
            hl_metrics = compute_metrics(y_val_hl, y_pred_hl)
            hl_metrics['Model'] = 'Hybrid_LSTM'
            results.append(hl_metrics)
        
        # 6. Hybrid TCN
        y_val_ht, y_pred_ht = eval_hybrid_model(target, 'tcn')
        if y_val_ht is not None:
            ht_metrics = compute_metrics(y_val_ht, y_pred_ht)
            ht_metrics['Model'] = 'Hybrid_TCN'
            results.append(ht_metrics)
        
        
        # Helper to sample predictions
        def sample_preds(y_true, y_pred, n=200):
            idx = np.random.choice(len(y_true), min(n, len(y_true)), replace=False)
            return {
                'actual': [float(x) for x in y_true[idx]],
                'predicted': [float(x) for x in y_pred[idx]]
            }
            
        prediction_samples[target] = {
            'V1 (XGBoost Tabular)': sample_preds(y_val_seq, y_pred_v1),
            'LSTM': sample_preds(y_val_lstm, y_pred_lstm),
            'TCN': sample_preds(y_val_tcn, y_pred_tcn),
            'Hybrid_LSTM': sample_preds(y_val_hl, y_pred_hl) if y_val_hl is not None else None,
            'Hybrid_TCN': sample_preds(y_val_ht, y_pred_ht) if y_val_ht is not None else None
        }
        
        all_results[target] = results
        
        # Print table for this target
        base_r2 = base_metrics['R2']
        print(f"{'Model':<25} | {'R2':>8} | {'MAE':>8} | {'RMSE':>8} | {'Lift':>8}")
        print("-" * 70)
        for r in results:
            lift = r['R2'] - base_r2
            print(f"{r['Model']:<25} | {r['R2']:>8.4f} | {r['MAE']:>8.4f} | {r['RMSE']:>8.4f} | {lift:>+8.4f}")
    
    # Save full results
    out_path = os.path.join(MODEL_DIR, 'full_paradigm_results.json')
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
        
    # Save a sample of predictions for UI plotting
    samples_path = os.path.join(MODEL_DIR, 'prediction_samples.json')
    with open(samples_path, 'w') as f:
        json.dump(prediction_samples, f, indent=2)
    
    print(f"\n\nFull results saved to {out_path}")
    print(f"Prediction samples saved to {samples_path}")
    print("\n" + "=" * 100)
    print("Phase 5 COMPLETE")
    print("=" * 100)

if __name__ == '__main__':
    main()
