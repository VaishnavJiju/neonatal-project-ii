"""
Temporal Intelligence Lab — Phase 2.5: Comprehensive Model Evaluation
=====================================================================
Evaluates all LSTM/TCN models against clinical baselines and Phase 2 scores.
"""
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os, json, pickle
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupShuffleSplit

# ============================================================
# MODEL DEFINITIONS (must match training)
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
            d = 2**i; ic = input_size if i==0 else nc[i-1]; p = (ks-1)*d
            layers.append(TCNBlock(ic, nc[i], ks, 1, d, p))
        self.tcn = nn.Sequential(*layers)
        self.fc = nn.Linear(nc[-1], 1)
    def forward(self, x):
        x = x.transpose(1, 2)
        return self.fc(self.tcn(x)[:, :, -1])

# ============================================================
# CONFIG
# ============================================================
DATA_DIR = 'mal_ed_data/multi_targets/v2'
MODEL_DIR = 'models/v2'

with open(os.path.join(DATA_DIR, 'metadata.json'), 'r') as f:
    META = json.load(f)
SEQ_LEN = META['window_size']
N_FEATURES = META['features_per_timestep']

# Phase 2 baseline scores (from previous evaluation at 15 features)
PHASE2_SCORES = {
    'delta_baz': {'LSTM': {'R2': 0.1602, 'MAE': 0.2669, 'RMSE': 0.3939},
                  'TCN':  {'R2': 0.1544, 'MAE': 0.2679, 'RMSE': 0.3953}},
    'illness':   {'LSTM': {'R2': 0.5311, 'MAE': 1.6387, 'RMSE': 3.9436},
                  'TCN':  {'R2': 0.5208, 'MAE': 1.6791, 'RMSE': 3.9864}},
}

# ============================================================
# EVALUATION
# ============================================================

def evaluate_target(target):
    df = pd.read_parquet(os.path.join(DATA_DIR, f'{target}_sequences.parquet'))
    feat_cols = [f'f_{i}' for i in range(SEQ_LEN * N_FEATURES)]
    X = df[feat_cols].values
    y = df['y'].values
    pids = df['pid'].values
    
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    _, val_idx = next(gss.split(X, y, groups=pids))
    X_val, y_val = X[val_idx], y[val_idx]
    
    # Load scaler
    with open(os.path.join(MODEL_DIR, f'scaler_{target}.pkl'), 'rb') as f:
        scaler = pickle.load(f)
    X_val_3d = X_val.reshape(-1, SEQ_LEN, N_FEATURES)
    X_val_scaled = scaler.transform(X_val_3d.reshape(-1, N_FEATURES)).reshape(-1, SEQ_LEN, N_FEATURES)
    X_val_t = torch.tensor(X_val_scaled, dtype=torch.float32)
    
    # Baselines
    if target == 'delta_baz':
        y_base = np.zeros_like(y_val)
        base_name = "Zero Growth"
    elif target == 'illness':
        # illness_days is feature index 7 in the last timestep
        idx = (SEQ_LEN - 1) * N_FEATURES + 7
        y_base = X_val[:, idx]
        base_name = "Persistence"
    elif target == 'recovery':
        # time_delta is feature index 1 in the last timestep
        idx = (SEQ_LEN - 1) * N_FEATURES + 1
        y_base = X_val[:, idx]
        base_name = "Last Interval"
    
    results = []
    
    for m_type in ['lstm', 'tcn']:
        model = LSTMModel(N_FEATURES) if m_type == 'lstm' else TCNModel(N_FEATURES)
        model.load_state_dict(torch.load(os.path.join(MODEL_DIR, f'{m_type}_{target}.pt'), map_location='cpu'))
        model.eval()
        
        with torch.no_grad():
            y_pred = model(X_val_t).numpy().flatten()
        
        mse = mean_squared_error(y_val, y_pred)
        results.append({
            'Model': m_type.upper(),
            'R2': r2_score(y_val, y_pred),
            'MAE': mean_absolute_error(y_val, y_pred),
            'RMSE': np.sqrt(mse),
        })
    
    # Baseline
    results.append({
        'Model': f'BASELINE ({base_name})',
        'R2': r2_score(y_val, y_base),
        'MAE': mean_absolute_error(y_val, y_base),
        'RMSE': np.sqrt(mean_squared_error(y_val, y_base)),
    })
    
    return results

# ============================================================
# MAIN
# ============================================================
all_results = {}
for target in ['delta_baz', 'illness', 'recovery']:
    all_results[target] = evaluate_target(target)

# Print combined table
print("=" * 100)
print(f"{'Target':<15} | {'Model':<22} | {'R2':>8} | {'MAE':>8} | {'RMSE':>8} | {'Phase 2 R2':>10} | {'Delta R2':>8}")
print("-" * 100)

for target, results in all_results.items():
    for r in results:
        model = r['Model']
        p2_r2 = ''
        delta_r2 = ''
        if target in PHASE2_SCORES and model in PHASE2_SCORES[target]:
            p2_val = PHASE2_SCORES[target][model]['R2']
            p2_r2 = f"{p2_val:>10.4f}"
            delta_r2 = f"{r['R2'] - p2_val:>+8.4f}"
        print(f"{target:<15} | {model:<22} | {r['R2']:>8.4f} | {r['MAE']:>8.4f} | {r['RMSE']:>8.4f} | {p2_r2:>10} | {delta_r2:>8}")
    print("-" * 100)

# Save results as JSON
with open(os.path.join(MODEL_DIR, 'evaluation_results.json'), 'w') as f:
    json.dump(all_results, f, indent=2)
print(f"\nResults saved to {MODEL_DIR}/evaluation_results.json")
